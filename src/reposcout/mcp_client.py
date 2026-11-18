"""Connects to the MCP servers and keeps one long-lived session per server.

`MultiServerMCPClient.get_tools()` opens a new session for every tool call by default.
For a stdio server that means starting a process per call. Measured on the codebase
server: one grep through a fresh session 637 ms, through a long-lived one 33 ms (Docker
mode would be slower still). So the API opens each session once, in its lifespan, and
every request reuses it. The sessions use anyio task groups, so they must be opened and
closed in the same task: no `asyncio.wait_for` around startup, no closing from another task.

A server that fails to start doesn't stop the app: it's reported in /health and the
agent works with the tools that are there (graceful degradation).
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .config import DEFAULT_CODEBASE_TOOLS, Settings
from .tool_policy import ToolSpec, is_read_only

log = logging.getLogger(__name__)

# Passed through to the server subprocesses (company proxies, custom CA bundles, Docker).
_PASSTHROUGH_ENV = [
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "GIT_SSL_CAINFO",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
]


def _passthrough() -> dict[str, str]:
    return {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}


def resolve_command(command: str) -> str:
    """`tools/github-mcp-server.exe` (relative, with a folder) is resolved against the working directory."""
    path = Path(command)
    if not path.is_absolute() and len(path.parts) > 1:
        return str(Path.cwd() / path)
    return command


def _session_kwargs(settings: Settings) -> dict[str, Any]:
    """A request that gets no answer fails after this long instead of waiting forever
    (a stuck server, a hung subprocess inside it). Longer than the per-call tool timeout,
    so the agent's own, friendlier timeout normally fires first."""
    return {"read_timeout_seconds": timedelta(seconds=settings.tool_timeout_s + 30)}


def github_connection(settings: Settings) -> dict[str, Any] | None:
    if not settings.github_token:
        return None
    tools = ",".join(settings.github_tools)
    env = {
        **_passthrough(),
        "GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token,
        "GITHUB_READ_ONLY": "1",  # the server drops every write tool, whatever else is configured
        "GITHUB_TOOLS": tools,  # only these tools are registered at all
    }
    mode = settings.github_mcp_mode
    if mode == "stdio":
        return {
            "transport": "stdio",
            "command": resolve_command(settings.github_mcp_command),
            "args": ["stdio"],
            "env": env,
            "session_kwargs": _session_kwargs(settings),
        }
    if mode == "docker":
        args = ["run", "-i", "--rm"]
        for name in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_READ_ONLY", "GITHUB_TOOLS"):
            args += [
                "-e",
                name,
            ]  # value comes from the env below, so the token isn't on the command line
        return {
            "transport": "stdio",
            "command": "docker",
            "args": [*args, settings.github_mcp_image, "stdio"],
            "env": env,
            "session_kwargs": _session_kwargs(settings),
        }
    # remote (api.githubcopilot.com) or a self-hosted `github-mcp-server http`
    return {
        "transport": "streamable_http",
        "url": settings.github_mcp_url,
        "headers": {
            "Authorization": f"Bearer {settings.github_token}",
            "X-MCP-Readonly": "true",
            "X-MCP-Tools": tools,
        },
        "session_kwargs": _session_kwargs(settings),
    }


def codebase_connection(settings: Settings) -> dict[str, Any] | None:
    if not settings.codebase_enabled:
        return None
    env = {
        **_passthrough(),
        "CODEBASE_CACHE_DIR": str(Path(settings.codebase_cache_dir).resolve()),
        "CODEBASE_GIT_BASE_URL": settings.codebase_git_base_url,
        "CODEBASE_CACHE_TTL_S": str(settings.codebase_cache_ttl_s),
        "CODEBASE_MAX_REPO_MB": str(settings.codebase_max_repo_mb),
        "CODEBASE_CLONE_TIMEOUT_S": str(settings.codebase_clone_timeout_s),
    }
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "reposcout.codebase.server"],
        "env": env,
        "session_kwargs": _session_kwargs(settings),
    }


@dataclass
class ServerStatus:
    name: str
    ok: bool
    transport: str = ""
    tools: list[str] = field(default_factory=list)
    error: str | None = None


def redact(text: str, settings: Settings) -> str:
    for secret in (settings.github_token, settings.llm_api_key):
        if secret:
            text = text.replace(secret, "***")
    return text


class McpToolbox:
    """Owns the MCP sessions and the registry of tools the agent may call."""

    def __init__(
        self, settings: Settings, connections: dict[str, dict[str, Any] | None] | None = None
    ):
        self.settings = settings
        self.connections = (
            connections
            if connections is not None
            else {
                "codebase": codebase_connection(settings),
                "github": github_connection(settings),
            }
        )
        self.specs: dict[str, ToolSpec] = {}
        self.status: dict[str, ServerStatus] = {}
        self._stacks: list[AsyncExitStack] = []

    def _allowed(self, server: str) -> set[str]:
        if server == "github":
            return set(self.settings.github_tools)
        if server == "codebase":
            return set(DEFAULT_CODEBASE_TOOLS)
        return set()  # unknown servers must be allow-listed explicitly

    async def start(self) -> None:
        active = {k: v for k, v in self.connections.items() if v}
        for name, conn in self.connections.items():
            if not conn:
                reason = "set GITHUB_TOKEN to enable" if name == "github" else "disabled"
                self.status[name] = ServerStatus(name, ok=False, error=reason)
        client = MultiServerMCPClient(active)  # type: ignore[arg-type]
        for name, conn in active.items():
            stack = AsyncExitStack()
            try:
                session = await stack.enter_async_context(client.session(name))
                tools = await load_mcp_tools(session, server_name=name)
            except Exception as exc:  # noqa: BLE001 - one broken server must not stop the app
                await stack.aclose()
                message = redact(f"{type(exc).__name__}: {exc}", self.settings)
                log.warning("mcp_server_unavailable server=%s error=%s", name, message)
                self.status[name] = ServerStatus(
                    name, ok=False, transport=conn["transport"], error=message[:300]
                )
                continue
            self._stacks.append(stack)
            registered = self.register(name, tools)
            self.status[name] = ServerStatus(
                name, ok=True, transport=conn["transport"], tools=registered
            )
            log.info("mcp_server_ready server=%s tools=%s", name, registered)

    def register(self, server: str, tools: list[Any]) -> list[str]:
        allowed = self._allowed(server)
        registered = []
        for tool in tools:
            if tool.name not in allowed:
                continue
            if not is_read_only(tool):
                log.warning("tool_skipped_not_read_only server=%s tool=%s", server, tool.name)
                continue
            if tool.name in self.specs:
                log.warning("tool_name_conflict server=%s tool=%s", server, tool.name)
                continue
            schema = (
                tool.args_schema
                if isinstance(tool.args_schema, dict)
                else tool.tool_call_schema.model_json_schema()
            )
            self.specs[tool.name] = ToolSpec(
                name=tool.name, server=server, tool=tool, schema=schema
            )
            registered.append(tool.name)
        return registered

    def has_server(self, server: str) -> bool:
        return any(spec.server == server for spec in self.specs.values())

    def llm_tools(self) -> list[dict[str, Any]]:
        return [spec.openai_spec for spec in self.specs.values()]

    async def aclose(self) -> None:
        while self._stacks:
            stack = self._stacks.pop()
            try:
                await stack.aclose()
            except Exception as exc:  # noqa: BLE001 - shutting down anyway
                log.debug("mcp_close_error %s", exc)