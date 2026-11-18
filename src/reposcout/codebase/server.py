"""The `codebase` MCP server: read-only tools over a shallow clone of a public GitHub repository.

Run it on its own (any MCP client can use it, e.g. Claude Desktop or the MCP Inspector):

    uv run python -m reposcout.codebase.server

The agent starts it as a stdio subprocess. Tools take `owner` and `repo`; the agent fills
those in itself, so the LLM never chooses which repository is read.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..config import get_settings
from . import core

READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
)


def build_server(cache: core.RepoCache) -> FastMCP:
    mcp = FastMCP(
        "codebase",
        instructions=(
            "Read-only access to the files of a public GitHub repository. Typical flow: "
            "repo_overview -> grep_code to locate -> read_file on the matching lines."
        ),
    )

    async def _checkout(owner: str, repo: str) -> core.Checkout:
        try:
            return await cache.checkout(owner, repo)
        except core.CodebaseError:
            raise
        except Exception as exc:  # unexpected: keep the message short for the model
            raise core.CodebaseError(
                f"Could not open {owner}/{repo}: {type(exc).__name__}: {str(exc)[:200]}"
            ) from exc

    @mcp.tool(annotations=READ_ONLY)
    async def repo_overview(owner: str, repo: str) -> str:
        """Overview of the repository: main languages, top-level folders, build/test/CI files and the
        start of the README. Use it first for questions about the project as a whole."""
        checkout = await _checkout(owner, repo)
        return await asyncio.to_thread(core.overview, checkout)

    @mcp.tool(annotations=READ_ONLY)
    async def list_files(
        owner: str, repo: str, path: str = "", depth: int = 2, pattern: str = ""
    ) -> str:
        """List files and folders under `path` (default: repository root), `depth` levels deep
        (1-4). Every entry is a full path from the repository root; folders end with "/".
        To find files of one kind anywhere in the repository, use `pattern`, e.g. "*.py"."""
        checkout = await _checkout(owner, repo)
        return await asyncio.to_thread(core.list_files, checkout, path, depth, pattern)

    @mcp.tool(annotations=READ_ONLY)
    async def grep_code(
        owner: str,
        repo: str,
        query: str,
        path: str = "",
        regex: bool = False,
        max_results: int = 40,
    ) -> str:
        """Search the code and return matching lines as `file:line: text` (case-insensitive).
        Search for one distinctive word or name, e.g. "faithfulness" or "class Session".
        To find several words at once, separate them with |, e.g. "faithfulness|precision|recall".
        `path` limits the search to a folder or file. `regex=true` only for a regular expression."""
        checkout = await _checkout(owner, repo)
        return await asyncio.to_thread(core.grep, checkout, query, path, regex, False, max_results)

    @mcp.tool(annotations=READ_ONLY)
    async def read_file(
        owner: str, repo: str, path: str, start_line: int = 1, end_line: int = 0
    ) -> str:
        """Read a file with line numbers, at most 200 lines per call. After grep_code found a match on
        line N, read around it, e.g. start_line=N-20, end_line=N+40."""
        checkout = await _checkout(owner, repo)
        return await asyncio.to_thread(core.read_file, checkout, path, start_line, end_line or None)

    return mcp


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=logging.WARNING)  # stdout is the MCP channel; logs go to stderr
    cache = core.RepoCache(
        core.CodebaseConfig(
            cache_dir=Path(settings.codebase_cache_dir).resolve(),
            git_base_url=settings.codebase_git_base_url,
            ttl_s=settings.codebase_cache_ttl_s,
            max_repo_mb=settings.codebase_max_repo_mb,
            clone_timeout_s=settings.codebase_clone_timeout_s,
        )
    )
    build_server(cache).run("stdio")


if __name__ == "__main__":
    main()