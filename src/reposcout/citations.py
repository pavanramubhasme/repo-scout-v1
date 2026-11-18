"""Sources are built from the tool calls that actually ran, not from what the LLM writes.

A citation the model invents is worthless; one derived from a successful tool call is
evidence. Each source is a link the user can click to check the answer.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from .repo_ref import RepoRef

_GREP_LINE = re.compile(r"^(?P<path>[^\s:][^:\n]*):(?P<line>\d+): ", re.MULTILINE)
_READ_HEADER = re.compile(r"\] (?P<path>\S+) lines (?P<start>\d+)-(?P<end>\d+) of \d+")


def sources_for_call(
    name: str, args: dict[str, Any], repo: RepoRef, output: str, max_grep_files: int = 3
) -> list[dict[str, str]]:
    base = repo.html_url
    blob = f"{base}/blob/HEAD"
    sources: list[dict[str, str]] = []

    def add(label: str, url: str, kind: str) -> None:
        sources.append({"label": label, "url": url, "kind": kind})

    if name == "read_file":
        match = _READ_HEADER.search(output)
        if match:
            path, start, end = match["path"], match["start"], match["end"]
            add(f"{path}:L{start}-L{end}", f"{blob}/{quote(path)}#L{start}-L{end}", "code")
    elif name == "grep_code":
        seen: list[tuple[str, str]] = []
        for match in _GREP_LINE.finditer(output):
            if match["path"] not in {p for p, _ in seen}:
                seen.append((match["path"], match["line"]))
        for path, line in seen[:max_grep_files]:
            add(f"{path}:L{line}", f"{blob}/{quote(path)}#L{line}", "code")
    elif name == "list_files":
        path = (args.get("path") or "").strip("/")
        add(f"{path or repo.name}/", f"{base}/tree/HEAD/{quote(path)}" if path else base, "code")
    elif name == "repo_overview":
        add(repo.full_name, base, "repo")
    elif name == "issue_read" and args.get("issue_number"):
        add(f"issue #{args['issue_number']}", f"{base}/issues/{args['issue_number']}", "issue")
    elif name == "pull_request_read" and args.get("pullNumber"):
        add(f"PR #{args['pullNumber']}", f"{base}/pull/{args['pullNumber']}", "pull_request")
    elif name == "get_commit" and args.get("sha"):
        add(f"commit {str(args['sha'])[:7]}", f"{base}/commit/{quote(str(args['sha']))}", "commit")
    elif name == "list_issues":
        add("issues", f"{base}/issues", "issue")
    elif name == "search_issues":
        add(
            f"issue search: {args.get('query', '')}",
            f"{base}/issues?q={quote(str(args.get('query', '')))}",
            "issue",
        )
    elif name == "list_pull_requests":
        add("pull requests", f"{base}/pulls", "pull_request")
    elif name == "list_commits":
        add("commits", f"{base}/commits", "commit")
    elif name in ("list_releases", "get_latest_release"):
        add("releases", f"{base}/releases", "release")
    return sources


def dedupe(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique = []
    for source in sources:
        if source["url"] not in seen:
            seen.add(source["url"])
            unique.append(source)
    return unique
