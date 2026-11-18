"""The codebase tools on a real (local) git clone."""

import asyncio
import json
import os
import time

import pytest

from reposcout.codebase import core
from tests.conftest import FAKE_TOKEN


@pytest.fixture
async def checkout(cache):
    return await cache.checkout("acme", "widgets")


async def test_clone_and_cache_reuse(cache):
    first = await cache.checkout("acme", "widgets")
    assert (first.root / "README.md").exists()
    assert len(first.sha) == 40 and first.branch == "main"
    marker = first.root / "src" / "widgets" / "core.py"
    mtime = marker.stat().st_mtime
    second = await cache.checkout("ACME", "Widgets")  # GitHub names are case-insensitive
    assert second.root == first.root
    assert marker.stat().st_mtime == mtime  # served from the cache, not cloned again


async def test_expired_cache_is_recloned(cache):
    first = await cache.checkout("acme", "widgets")
    pointer = cache.config.cache_dir / "acme__widgets.json"
    data = json.loads(pointer.read_text())
    data["cloned_at"] = time.time() - 10 * cache.config.ttl_s
    pointer.write_text(json.dumps(data))
    second = await cache.checkout("acme", "widgets")
    assert json.loads(pointer.read_text())["cloned_at"] > data["cloned_at"]
    assert second.sha == first.sha
    assert second.root != first.root and not first.root.exists()  # old clone removed


async def test_missing_repo_fails_fast(cache):
    with pytest.raises(core.CodebaseError, match="not found|failed"):
        await cache.checkout("acme", "does-not-exist")
    assert list(cache.config.cache_dir.iterdir()) == []  # nothing half-cloned left behind


async def test_invalid_name_is_rejected_before_git_runs(cache):
    with pytest.raises(core.CodebaseError):
        await cache.checkout("acme", "../../etc")


async def test_too_large_repo_is_rejected(tmp_path, git_remote):
    small = core.RepoCache(
        core.CodebaseConfig(cache_dir=tmp_path / "c", git_base_url=git_remote, max_repo_mb=0)
    )
    with pytest.raises(core.CodebaseError, match="limit"):
        await small.checkout("acme", "widgets")


def test_overview(checkout):
    text = core.overview(checkout)
    assert "[acme/widgets @ main" in text
    assert "Python" in text and "pyproject.toml" in text
    assert ".github/workflows (1 CI workflows: ci.yml)" in text
    assert "tests/" in text and "LICENSE" in text
    assert "A tiny library that makes widgets." in text


def test_list_files(checkout):
    text = core.list_files(checkout, "src", depth=3)
    assert "widgets/" in text and "core.py" in text
    root = core.list_files(checkout)
    assert ".git/" not in root
    with pytest.raises(core.CodebaseError, match="No such directory"):
        core.list_files(checkout, "nope")


def test_grep_literal_by_default(checkout):
    text = core.grep(checkout, "Widget::make")  # would be odd as a regex; literal by default
    assert "src/widgets/core.py:5:" in text


def test_grep_case_insensitive_and_sources_first(checkout):
    text = core.grep(checkout, "make_widget")
    lines = text.splitlines()[1:]
    assert lines[0].startswith("src/widgets/")  # definitions before tests
    assert any(line.startswith("tests/test_core.py") for line in lines)


def test_grep_regex_path_and_limits(checkout):
    assert "MAX_WIDGETS = 42" in core.grep(checkout, r"MAX_\w+ = \d+", regex=True)
    assert "No matches" in core.grep(checkout, "make_widget", path="README.md")
    many = core.grep(checkout, "filler line", max_results=5)
    assert "300 matches" in many and "first 5 shown" in many
    with pytest.raises(core.CodebaseError, match="Invalid regular expression"):
        core.grep(checkout, "(", regex=True)
    with pytest.raises(core.CodebaseError):
        core.grep(checkout, "")


def test_read_file_ranges(checkout):
    text = core.read_file(checkout, "src/widgets/core.py", 4, 8)
    assert "lines 4-8 of" in text
    assert "    4| def make_widget(name: str) -> dict:" in text
    whole = core.read_file(checkout, "src/widgets/core.py")
    assert "lines 1-200 of" in whole and "call read_file with start_line=201" in whole


def test_read_file_accepts_what_models_send(checkout):
    for path in (
        "./src/widgets/core.py",
        "/src/widgets/core.py",
        "src\\widgets\\core.py",
        "https://github.com/acme/widgets/blob/main/src/widgets/core.py#L3",
    ):
        assert "MAX_WIDGETS" in core.read_file(checkout, path, 1, 3)


def test_read_file_suggests_the_right_path(checkout):
    with pytest.raises(core.CodebaseError, match="Did you mean: src/widgets/core.py"):
        core.read_file(checkout, "core.py")


@pytest.mark.parametrize(
    "path", ["../../../etc/passwd", "/../../etc/passwd", ".git/config", "src/../../x"]
)
def test_read_file_cannot_escape_the_clone(checkout, path):
    with pytest.raises(core.CodebaseError):
        core.read_file(checkout, path)


def test_binary_and_directories(checkout):
    with pytest.raises(core.CodebaseError, match="binary"):
        core.read_file(checkout, "logo.png")
    with pytest.raises(core.CodebaseError, match="directory"):
        core.read_file(checkout, "src")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_symlinks_are_not_followed(checkout):
    text = core.read_file(checkout, "leak")
    assert "/etc/passwd" in text  # the link target as text...
    assert "root:" not in text  # ...never the file it points to


def test_secrets_are_redacted(checkout):
    assert FAKE_TOKEN not in core.read_file(checkout, "src/widgets/settings.py")
    assert "[REDACTED_SECRET]" in core.grep(checkout, "API_TOKEN")
    assert core.redact_secrets("key AKIAABCDEFGHIJKLMNOP end") == "key [REDACTED_SECRET] end"


async def test_git_never_inherits_stdin(monkeypatch):
    """Regression (Windows): in the stdio MCP server, stdin is the protocol pipe. A git child
    that inherited it blocked at startup, so the first clone through the server hung forever."""
    seen = {}
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        return await real(*args, **kwargs)

    monkeypatch.setattr(core.asyncio, "create_subprocess_exec", spy)
    assert "git version" in await core._git(["--version"])
    assert seen["stdin"] == asyncio.subprocess.DEVNULL


async def test_a_locked_old_clone_never_blocks_a_refresh(cache, monkeypatch):
    """Regression (Windows): a virus scanner holds a fresh clone's files open, so renaming
    or deleting the folder fails with 'Access is denied'. No folder is ever renamed, and a
    folder that can't be deleted yet is left for a later cleanup."""
    first = await cache.checkout("acme", "widgets")
    pointer = cache.config.cache_dir / "acme__widgets.json"
    data = json.loads(pointer.read_text())
    data["cloned_at"] = 0
    pointer.write_text(json.dumps(data))

    def locked(path, **_kwargs):
        raise PermissionError(f"[WinError 5] Access is denied: {path}")

    monkeypatch.setattr(core.shutil, "rmtree", locked)
    monkeypatch.setattr(core.time, "sleep", lambda _s: None)
    second = await cache.checkout("acme", "widgets")
    assert second.root != first.root and (second.root / "README.md").exists()
    assert first.root.exists()  # deletion deferred, not fatal
    assert json.loads(pointer.read_text())["dir"] == second.root.name


async def test_an_unfinished_clone_is_ignored(cache):
    stale = cache.config.cache_dir / "acme__widgets-deadbeef"
    (stale / ".git").mkdir(parents=True)  # a clone that crashed before its pointer was written
    checkout = await cache.checkout("acme", "widgets")
    assert checkout.root != stale and (checkout.root / "README.md").exists()


async def test_cache_keeps_at_most_max_repos(tmp_path, git_remote):
    small = core.RepoCache(
        core.CodebaseConfig(cache_dir=tmp_path / "c", git_base_url=git_remote, max_repos=1)
    )
    first = await small.checkout("acme", "widgets")
    pointer = small.config.cache_dir / "acme__widgets.json"
    other = small.config.cache_dir / "other__repo.json"
    other.write_text(json.dumps({"dir": "other__repo-1", "cloned_at": 1}))
    (small.config.cache_dir / "other__repo-1").mkdir()
    small._cleanup()
    assert pointer.exists() and first.root.exists()
    assert not other.exists() and not (small.config.cache_dir / "other__repo-1").exists()


def test_grep_pipe_means_any_of_without_regex(checkout):
    """Regression: a 7B model searched "faithfulness|precision|recall" with regex=false and got
    0 matches on a repo that uses all three words, then told the user they weren't used."""
    text = core.grep(checkout, "MAX_WIDGETS|RETRIES")
    assert "for any of: MAX_WIDGETS, RETRIES" in text
    assert "src/widgets/core.py:1:" in text and "src/widgets/settings.py:2:" in text


def test_grep_phrase_without_exact_match_falls_back_to_its_words(checkout):
    text = core.grep(checkout, "widget limit retries")
    assert "no line contains the exact text" in text and "RETRIES = 3" in text
    assert "No matches" in core.grep(
        checkout, "zebra giraffe"
    ) and "any of: zebra, giraffe" in core.grep(checkout, "zebra giraffe")
    assert "No matches" in core.grep(checkout, "the and for")  # only stop words: no fallback


def test_list_files_shows_full_paths_so_folders_are_unambiguous(checkout):
    """Regression: an indented tree printed each folder's files without the folder name, and a
    7B model listed eval/run_eval.py as ragapp/run_eval.py. Every entry is now a full path."""
    lines = core.list_files(checkout, depth=3).splitlines()[1:]
    assert (
        "src/widgets/core.py" in lines
        and "tests/test_core.py" in lines
        and ".github/workflows/" in lines
    )
    assert all(not line.startswith(" ") for line in lines)
    assert lines.index("src/") < lines.index("src/widgets/") < lines.index("src/widgets/core.py")


def test_list_files_pattern_finds_files_at_any_depth(checkout):
    text = core.list_files(checkout, pattern="*.py")
    assert "4 files matching '*.py'" in text
    lines = text.splitlines()[1:]
    assert lines == [
        "src/widgets/__init__.py",
        "src/widgets/core.py",
        "src/widgets/settings.py",
        "tests/test_core.py",
    ]
    assert core.list_files(checkout, "tests", pattern=".py").splitlines()[1:] == [
        "tests/test_core.py"
    ]
    assert "none" in core.list_files(checkout, pattern="*.rs")