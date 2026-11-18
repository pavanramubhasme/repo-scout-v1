"""Read-only access to a shallow clone of a GitHub repository.

Why our own server next to GitHub's: reading code through the GitHub API is chatty
and unbounded (a whole file per call, code search has its own low rate limit and no
regex). A local shallow clone gives fast grep and line-range reads, so a small LLM
gets exactly the 40 lines it needs instead of a 3,000-line file.

Everything here is plain Python (no MCP) so it is easy to test; `server.py` wraps it.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import re
import shutil
import stat
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..repo_ref import InvalidRepo, parse_repo

log = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git",
    "node_modules",
    "vendor",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "target",
    ".next",
    ".idea",
    ".vscode",
    "site-packages",
    ".gradle",
    "coverage",
    "htmlcov",
}
MAX_FILE_BYTES = 1_000_000
MAX_READ_LINES = 200
MAX_LINE_CHARS = 300
GREP_TIME_BUDGET_S = 10.0

LANGUAGES = {
    ".py": "Python",
    ".pyi": "Python",
    ".ipynb": "Jupyter",
    ".js": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".scala": "Scala",
    ".c": "C",
    ".h": "C/C++ header",
    ".cc": "C++",
    ".cpp": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".m": "Objective-C",
    ".dart": "Dart",
    ".lua": "Lua",
    ".r": "R",
    ".jl": "Julia",
    ".sh": "Shell",
    ".ps1": "PowerShell",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".md": "Markdown",
    ".rst": "reStructuredText",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".toml": "TOML",
    ".json": "JSON",
    ".tf": "Terraform",
    ".proto": "Protobuf",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".erl": "Erlang",
    ".hs": "Haskell",
    ".ml": "OCaml",
    ".zig": "Zig",
    ".nim": "Nim",
}
MANIFESTS = {
    "pyproject.toml": "Python project (pyproject)",
    "setup.py": "Python package (setup.py)",
    "requirements.txt": "Python requirements",
    "Pipfile": "Pipenv",
    "uv.lock": "uv lock file",
    "poetry.lock": "Poetry lock file",
    "package.json": "Node.js package",
    "go.mod": "Go module",
    "Cargo.toml": "Rust crate",
    "pom.xml": "Maven",
    "build.gradle": "Gradle",
    "build.gradle.kts": "Gradle (Kotlin)",
    "Gemfile": "Ruby bundle",
    "composer.json": "PHP Composer",
    "CMakeLists.txt": "CMake",
    "Makefile": "Makefile",
    "Dockerfile": "Docker image",
    "docker-compose.yml": "Docker Compose",
    "compose.yaml": "Docker Compose",
    "tox.ini": "tox",
    "noxfile.py": "nox",
    "pytest.ini": "pytest config",
    "setup.cfg": "setup.cfg",
    ".pre-commit-config.yaml": "pre-commit hooks",
    "mkdocs.yml": "MkDocs docs",
}
_SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{60,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
]


class CodebaseError(Exception):
    """A user-facing error (bad path, repo not found, ...). The message goes back to the model."""


def redact_secrets(text: str) -> str:
    """Mask credentials that people accidentally commit, so they never reach the LLM or the UI."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


@dataclass(frozen=True)
class CodebaseConfig:
    cache_dir: Path
    git_base_url: str = "https://github.com"
    ttl_s: int = 3600
    max_repo_mb: int = 300
    clone_timeout_s: int = 180
    max_repos: int = 20


@dataclass(frozen=True)
class Checkout:
    owner: str
    name: str
    root: Path
    sha: str
    branch: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


def _rmtree(path: Path) -> None:
    """Delete a directory, also on Windows where git pack files are read-only."""

    def _make_writable(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    for attempt in range(5):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=_make_writable)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))  # Windows: a virus scanner still holds a file


def _try_rmtree(path: Path) -> bool:
    """Best-effort delete. A clone still locked by a virus scanner stays until the next cleanup."""
    try:
        _rmtree(path)
        return True
    except OSError as exc:
        log.warning("cache_cleanup_deferred path=%s error=%s", path.name, exc)
        return False


def _write_json(path: Path, data: dict) -> None:
    """Write a small file atomically: temp file + os.replace (retried if briefly locked)."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                break
            time.sleep(0.25)
    # Still locked after ~2 s: fall back to writing in place (not atomic, but readers
    # treat a broken pointer as "no clone yet" and simply clone again).
    tmp.unlink(missing_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class RepoCache:
    """Shallow clones in a local cache, refreshed after `ttl_s`.

    Layout: every clone gets its own new folder (`psf__requests-1a2b3c4d/`) and the
    pointer file `psf__requests.json`, written last, says which folder is current.

    Why not clone to a temp folder and rename it into place: on Windows a virus scanner
    (or OneDrive) holds the files of a fresh clone open for a few seconds, and renaming or
    deleting a folder fails while any file in it is open ("Access is denied"). Here no
    folder is ever renamed, and deleting an old one is best-effort: if it's locked, it's
    removed at a later cleanup. A clone that never got its pointer file (crash, timeout)
    is simply ignored.
    """

    def __init__(self, config: CodebaseConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def _key(self, owner: str, name: str) -> str:
        return f"{owner}__{name}".lower()

    def _pointer(self, key: str) -> Path:
        return self.config.cache_dir / f"{key}.json"

    def _read_pointer(self, key: str) -> dict | None:
        try:
            meta = json.loads(self._pointer(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return meta if isinstance(meta, dict) and meta.get("dir") else None

    def _load(self, key: str, owner: str, name: str) -> Checkout | None:
        meta = self._read_pointer(key)
        if not meta or time.time() - meta.get("cloned_at", 0) > self.config.ttl_s:
            return None
        root = self.config.cache_dir / meta["dir"]
        if not (root / ".git").is_dir():
            return None
        return Checkout(owner, name, root, meta.get("sha", ""), meta.get("branch", ""))

    async def checkout(self, owner: str, name: str) -> Checkout:
        try:
            ref = parse_repo(f"{owner}/{name}")
        except InvalidRepo as exc:
            raise CodebaseError(str(exc)) from exc
        key = self._key(ref.owner, ref.name)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._load(key, ref.owner, ref.name)
            if cached:
                return cached
            return await self._clone(ref.owner, ref.name, key)

    async def _clone(self, owner: str, name: str, key: str) -> Checkout:
        root = self.config.cache_dir / f"{key}-{uuid.uuid4().hex[:8]}"
        url = f"{self.config.git_base_url.rstrip('/')}/{owner}/{name}.git"
        try:
            await _git(
                # No stored credentials and no prompts: a private or missing repo fails fast
                # instead of opening a login window (Git Credential Manager on Windows).
                # core.symlinks=false checks symlinks out as plain text files, so a link
                # in the repo can never point the reader at a file outside the clone.
                [
                    "-c",
                    "credential.helper=",
                    "-c",
                    "core.symlinks=false",
                    "clone",
                    "--depth",
                    "1",
                    "--single-branch",
                    "--no-tags",
                    url,
                    str(root),
                ],
                timeout=self.config.clone_timeout_s,
            )
            size_mb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e6
            if size_mb > self.config.max_repo_mb:
                raise CodebaseError(
                    f"{owner}/{name} is {size_mb:.0f} MB, above the {self.config.max_repo_mb} MB limit."
                )
            sha = (await _git(["rev-parse", "HEAD"], cwd=root)).strip()
            branch = (await _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)).strip()
        except BaseException:
            _try_rmtree(root)
            raise
        previous = self._read_pointer(key)
        _write_json(
            self._pointer(key),
            {"dir": root.name, "sha": sha, "branch": branch, "cloned_at": time.time()},
        )
        if previous and previous["dir"] != root.name:
            _try_rmtree(self.config.cache_dir / previous["dir"])
        self._cleanup()
        return Checkout(owner, name, root, sha, branch)

    def _cleanup(self) -> None:
        """Keep at most `max_repos` repositories; remove folders no pointer refers to
        (old clones that were locked earlier, crashed clones) once they're an hour old."""
        pointers = []
        for path in self.config.cache_dir.glob("*.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                pointers.append((meta.get("cloned_at", 0), path, meta.get("dir", "")))
            except (OSError, ValueError):
                continue
        pointers.sort()
        for _, path, folder in pointers[: max(0, len(pointers) - self.config.max_repos)]:
            path.unlink(missing_ok=True)
            _try_rmtree(self.config.cache_dir / folder)
        live = {folder for _, path, folder in pointers if path.exists()}
        cutoff = time.time() - 3600
        for folder in self.config.cache_dir.iterdir():
            if folder.is_dir() and folder.name not in live:
                try:
                    if folder.stat().st_mtime < cutoff:
                        _try_rmtree(folder)
                except OSError:
                    continue


async def _git(args: list[str], cwd: Path | None = None, timeout: float = 60) -> str:
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_LFS_SKIP_SMUDGE": "1",
    }
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=cwd,
            env=env,
            # Never let git inherit stdin: in the stdio MCP server, stdin is the protocol pipe.
            # On Windows a child that inherits a pipe with a pending read blocks at startup
            # (the server thread's ReadFile holds the handle), so every clone hung forever.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise CodebaseError("git is not installed on the server.") from exc
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise CodebaseError(f"git timed out after {timeout:.0f} s (repository too large?)") from exc
    if proc.returncode != 0:
        message = err.decode("utf-8", "replace")
        if any(
            s in message
            for s in (
                "not found",
                "could not read Username",
                "Authentication failed",
                "terminal prompts disabled",
            )
        ):
            raise CodebaseError(
                "Repository not found, or it is private (only public repositories can be cloned)."
            )
        raise CodebaseError(
            f"git failed: {message.strip().splitlines()[-1] if message.strip() else proc.returncode}"
        )
    return out.decode("utf-8", "replace")


# --- helpers -----------------------------------------------------------------


def _iter_files(root: Path, start: Path):
    """Yield files below `start`, skipping vendored/build directories, in a stable order."""
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_text(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
        with path.open("rb") as handle:
            return b"\0" not in handle.read(8192)
    except OSError:
        return False


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def normalize_path(path: str | None) -> str:
    """Accept what small models actually send: './x', '/x', 'x\\y', or a full github.com blob URL."""
    text = (path or "").strip().strip("'\"`")
    blob = re.match(r"^https?://github\.com/[^/]+/[^/]+/(?:blob|tree)/[^/]+/(.*)$", text)
    if blob:
        text = blob.group(1)
    text = text.split("#", 1)[0].replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _resolve(root: Path, path: str | None) -> Path:
    rel = normalize_path(path)
    if rel in ("", "."):
        return root
    if ".git" in rel.split("/"):
        raise CodebaseError("The .git directory is not readable.")
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise CodebaseError("Path is outside the repository.")
    return candidate


def _suggest(root: Path, path: str, limit: int = 5) -> str:
    name = Path(normalize_path(path)).name.lower()
    if not name:
        return ""
    hits = [_rel(root, f) for f in _iter_files(root, root) if f.name.lower() == name][:limit]
    return (
        f" Did you mean: {', '.join(hits)}?" if hits else " Use list_files or grep_code to find it."
    )


def _header(co: Checkout) -> str:
    return f"[{co.full_name} @ {co.branch} {co.sha[:7]}]"


# --- tools -------------------------------------------------------------------


def overview(co: Checkout, readme_lines: int = 40) -> str:
    root = co.root
    lines_by_lang: Counter[str] = Counter()
    files_by_lang: Counter[str] = Counter()
    total = 0
    tests_dirs: set[str] = set()
    for f in _iter_files(root, root):
        total += 1
        parts = _rel(root, f).split("/")
        for i, part in enumerate(parts[:-1]):
            if part in {"tests", "test", "__tests__", "spec"}:
                tests_dirs.add("/".join(parts[: i + 1]) + "/")
                break
        lang = LANGUAGES.get(f.suffix.lower())
        if lang and _is_text(f):
            files_by_lang[lang] += 1
            try:
                with f.open("rb") as handle:
                    lines_by_lang[lang] += sum(1 for _ in handle)
            except OSError:
                pass
    all_lines = sum(lines_by_lang.values()) or 1
    langs = ", ".join(
        f"{lang} {count * 100 / all_lines:.0f}% ({count:,} lines, {files_by_lang[lang]} files)"
        for lang, count in lines_by_lang.most_common(6)
    )
    top_level = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    top = "  ".join(p.name + ("/" if p.is_dir() else "") for p in top_level if p.name != ".git")
    manifests = [f"{name} ({label})" for name, label in MANIFESTS.items() if (root / name).exists()]
    workflows = root / ".github" / "workflows"
    if workflows.is_dir():
        names = sorted(p.name for p in workflows.iterdir() if p.suffix in (".yml", ".yaml"))
        manifests.append(f".github/workflows ({len(names)} CI workflows: {', '.join(names[:6])})")
    license_file = next(
        (p.name for p in top_level if p.name.upper().startswith(("LICENSE", "COPYING"))), None
    )
    readme = next(
        (p for p in top_level if p.is_file() and p.name.lower().startswith("readme")), None
    )
    parts = [
        f"{_header(co)} {total} files",
        f"Languages by lines: {langs or 'none detected'}",
        f"Top level: {top}",
        f"Build/test/CI files: {', '.join(manifests) or 'none found'}",
        f"Test locations: {', '.join(sorted(tests_dirs)[:5]) or 'none found'}",
        f"License file: {license_file or 'none found'}",
    ]
    if readme:
        head = _read_lines(readme)[:readme_lines]
        parts.append(
            f"{readme.name} (first {len(head)} lines):\n"
            + "\n".join(line[:MAX_LINE_CHARS] for line in head)
        )
    return redact_secrets("\n".join(parts))


def list_files(
    co: Checkout, path: str = "", depth: int = 2, pattern: str = "", max_entries: int = 300
) -> str:
    """Every entry is a full path from the repository root (`eval/run_eval.py`, folders end
    with `/`), sorted so a folder's contents follow it. An indented tree looked tidy but
    didn't say which folder a file was in: a 7B model put `eval/run_eval.py` under `ragapp/`.
    Full paths are unambiguous, and the model can pass them straight to read_file.

    `pattern` (e.g. "*.py") lists matching files at any depth."""
    start = _resolve(co.root, path)
    if not start.exists():
        raise CodebaseError(
            f"No such directory: {normalize_path(path)!r}.{_suggest(co.root, path)}"
        )
    if start.is_file():
        return f"{_header(co)} {_rel(co.root, start)} is a file ({start.stat().st_size:,} bytes). Use read_file."
    depth = max(1, min(int(depth or 2), 4))
    pattern = (pattern or "").strip()
    if pattern and not any(ch in pattern for ch in "*?["):
        pattern = f"*{pattern}*"  # ".py" or "test" work as "contains"
    entries: list[str] = []
    base_parts = len(start.relative_to(co.root).parts)
    truncated = False
    for dirpath, dirnames, filenames in os.walk(start):
        current = Path(dirpath)
        level = len(current.relative_to(co.root).parts) - base_parts
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        if pattern:
            entries += [
                _rel(co.root, current / f)
                for f in filenames
                if fnmatch.fnmatch(f.lower(), pattern.lower())
            ]
        else:
            if level >= depth:
                dirnames[:] = []
                continue
            entries += [_rel(co.root, current / d) + "/" for d in dirnames]
            entries += [_rel(co.root, current / f) for f in filenames]
        if len(entries) > max_entries:
            truncated = True
            break
    entries.sort(key=lambda e: [part.lower() for part in e.rstrip("/").split("/")])
    label = _rel(co.root, start) + "/" if start != co.root else "repository root"
    if pattern:
        title = f"{len(entries[:max_entries])} files matching {pattern!r} under {label}"
    else:
        folders = sum(1 for e in entries if e.endswith("/"))
        title = f"{label}, {depth} levels: {folders} folders, {len(entries) - folders} files"
    if not entries:
        return f"{_header(co)} {title}: none."
    note = "\n[... more entries; narrow with path=... or pattern=...]" if truncated else ""
    return f"{_header(co)} {title} (full paths)\n" + "\n".join(entries[:max_entries]) + note


_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "is",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "or",
    "with",
    "how",
    "what",
    "where",
    "which",
    "does",
    "do",
    "they",
    "this",
    "that",
    "used",
    "use",
    "using",
    "method",
    "methods",
}


def _search(co: Checkout, files: list[Path], pattern: re.Pattern, max_results: int):
    hits: list[str] = []
    matched_files: set[str] = set()
    total = 0
    deadline = time.monotonic() + GREP_TIME_BUDGET_S
    stopped = False
    for f in files:
        if time.monotonic() > deadline:
            stopped = True
            break
        if not _is_text(f):
            continue
        rel = _rel(co.root, f)
        try:
            lines = _read_lines(f)
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if pattern.search(line[:1000]):
                total += 1
                matched_files.add(rel)
                if len(hits) < max_results:
                    hits.append(f"{rel}:{number}: {line.strip()[:MAX_LINE_CHARS]}")
    return hits, total, matched_files, stopped


def _any_of(terms: list[str], flags: int) -> re.Pattern:
    return re.compile("|".join(re.escape(t) for t in terms), flags)


def grep(
    co: Checkout,
    query: str,
    path: str = "",
    regex: bool = False,
    case_sensitive: bool = False,
    max_results: int = 40,
) -> str:
    """Literal, case-insensitive search by default, made forgiving for small models:
    `a|b|c` means "any of these" even without regex=true, and a phrase with no exact match
    falls back to "any of its words" (the header says so, so the model knows)."""
    query = (query or "").strip()
    if not query:
        raise CodebaseError("query must not be empty.")
    if len(query) > 200:
        raise CodebaseError("query is too long (max 200 characters).")
    flags = 0 if case_sensitive else re.IGNORECASE
    alternatives = [t.strip() for t in query.split("|") if t.strip()]
    try:
        if regex:
            pattern = re.compile(query, flags)
        elif len(alternatives) > 1:
            pattern = _any_of(alternatives, flags)
        else:
            pattern = re.compile(re.escape(query), flags)
    except re.error as exc:
        raise CodebaseError(f"Invalid regular expression: {exc}. Try regex=false.") from exc
    start = _resolve(co.root, path)
    if not start.exists():
        raise CodebaseError(f"No such path: {normalize_path(path)!r}.{_suggest(co.root, path)}")
    files = [start] if start.is_file() else list(_iter_files(co.root, start))
    # Source files first, tests and docs last: definitions are usually what the question is about.
    files.sort(
        key=lambda f: (
            bool(re.search(r"(^|/)(tests?|docs?|examples?)(/|$)", _rel(co.root, f))),
            _rel(co.root, f),
        )
    )
    max_results = max(1, min(int(max_results or 40), 100))
    hits, total, matched_files, stopped = _search(co, files, pattern, max_results)
    note = f" for any of: {', '.join(alternatives)}" if not regex and len(alternatives) > 1 else ""

    words = [w for w in re.findall(r"[\w.-]+", query) if len(w) > 2 and w.lower() not in _STOPWORDS]
    if not hits and not regex and len(alternatives) == 1 and len(words) > 1:
        hits, total, matched_files, stopped = _search(co, files, _any_of(words, flags), max_results)
        note = (
            f" (no line contains the exact text {query!r}; lines with any of: {', '.join(words)})"
        )

    if not hits:
        scope = f" under {normalize_path(path)!r}" if normalize_path(path) else ""
        tried = f" or any of: {', '.join(words)}" if len(words) > 1 and not regex else ""
        return f"{_header(co)} No matches for {query!r}{tried}{scope}."
    summary = f"{_header(co)} {total} matches in {len(matched_files)} files{note}"
    if total > len(hits):
        summary += f" (first {len(hits)} shown; narrow with path=... or a more specific query)"
    if stopped:
        summary += " [search stopped at the time limit]"
    return redact_secrets(summary + "\n" + "\n".join(hits))


def read_file(co: Checkout, path: str, start_line: int = 1, end_line: int | None = None) -> str:
    target = _resolve(co.root, path)
    rel = normalize_path(path)
    if not target.exists():
        raise CodebaseError(f"No such file: {rel!r}.{_suggest(co.root, path)}")
    if target.is_dir():
        raise CodebaseError(f"{rel!r} is a directory. Use list_files.")
    if not _is_text(target):
        raise CodebaseError(f"{rel!r} is binary or larger than {MAX_FILE_BYTES // 1_000_000} MB.")
    lines = _read_lines(target)
    total = len(lines)
    if total == 0:
        return f"{_header(co)} {rel} is empty."
    start = max(1, int(start_line or 1))
    if start > total:
        raise CodebaseError(f"{rel} has only {total} lines.")
    end = int(end_line) if end_line else start + MAX_READ_LINES - 1
    end = min(max(end, start), total, start + MAX_READ_LINES - 1)
    body = "\n".join(f"{n:>5}| {lines[n - 1][:MAX_LINE_CHARS]}" for n in range(start, end + 1))
    footer = ""
    if end < total:
        footer = f"\n[... {total - end} more lines; call read_file with start_line={end + 1}]"
    return redact_secrets(
        f"{_header(co)} {_rel(co.root, target)} lines {start}-{end} of {total}\n{body}{footer}"
    )