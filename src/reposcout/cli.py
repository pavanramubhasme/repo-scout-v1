"""Command line: `uv run repo-scout ask psf/requests "How are retries configured?"`

repo-scout ask <repo> <question>   answer one question (starts the MCP servers itself)
repo-scout tools                   show which MCP servers and tools are available
repo-scout graph                   print the LangGraph workflow as Mermaid
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .agent import RepoAgent
from .config import get_settings
from .llm import build_chat_model
from .mcp_client import McpToolbox
from .repo_ref import parse_repo


async def _with_agent(fn):
    settings = get_settings()
    toolbox = McpToolbox(settings)
    await toolbox.start()
    try:
        return await fn(RepoAgent(build_chat_model(settings), toolbox, settings), toolbox)
    finally:
        await toolbox.aclose()


async def _ask(repo: str, question: str) -> int:
    ref = parse_repo(repo)

    async def run(agent: RepoAgent, _toolbox: McpToolbox) -> int:
        result = await agent.ask(ref, question)
        print(result.answer)
        print()
        for source in result.sources:
            print(f"  source: {source['label']}  {source['url']}")
        path = " -> ".join(s["tool"] if s["node"] == "tool" else s["node"] for s in result.steps)
        print(f"  path: {path}")
        failures = {
            (st["tool"], st.get("error", ""))
            for st in result.steps
            if st["node"] == "tool" and not st.get("ok")
        }
        for tool, error in sorted(failures):
            print(f"  failed: {tool}: {' '.join(error.split())[:240]}")
        print(
            f"  outcome: {result.outcome}, tool calls: {result.tool_calls}, {result.latency_ms / 1000:.1f} s"
        )
        return 0 if result.outcome == "answered" else 1

    return await _with_agent(run)


async def _tools() -> int:
    async def run(_agent: RepoAgent, toolbox: McpToolbox) -> int:
        for name, status in toolbox.status.items():
            state = "ok" if status.ok else f"unavailable ({status.error})"
            print(f"{name}: {state}")
            for tool in status.tools:
                print(f"  - {tool}")
        return 0 if toolbox.specs else 1

    return await _with_agent(run)


async def _graph() -> int:
    async def run(agent: RepoAgent, _toolbox: McpToolbox) -> int:
        print(agent.mermaid())
        return 0

    return await _with_agent(run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="repo-scout", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("repo", help="owner/name or a github.com URL")
    ask.add_argument("question")
    sub.add_parser("tools", help="list MCP servers and tools")
    sub.add_parser("graph", help="print the agent graph as Mermaid")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if args.command == "ask":
        return asyncio.run(_ask(args.repo, args.question))
    if args.command == "tools":
        return asyncio.run(_tools())
    return asyncio.run(_graph())


if __name__ == "__main__":
    raise SystemExit(main())