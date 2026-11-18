"""All settings come from environment variables (or a `.env` file)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Read-only GitHub tools the agent may use. Code reading goes through our own
# `codebase` server (bounded, line-numbered output), so the GitHub server only
# covers the API surface: issues, pull requests, commits and releases.
DEFAULT_GITHUB_TOOLS = [
    "list_issues",
    "issue_read",
    "search_issues",
    "list_pull_requests",
    "pull_request_read",
    "list_commits",
    "get_commit",
    "list_releases",
    "get_latest_release",
]

DEFAULT_CODEBASE_TOOLS = ["repo_overview", "list_files", "grep_code", "read_file"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -----------------------------------------------------------------
    llm_provider: Literal["ollama", "openai"] = "ollama"
    """`ollama` for local models; `openai` for any OpenAI-compatible API (Groq, OpenRouter, vLLM, OpenAI)."""
    llm_model: str = "granite4:3b"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_num_ctx: int = 16384
    """Ollama context window. Ollama's default is small: tool schemas + results overflow it silently."""
    llm_timeout_s: float = 120.0

    # --- GitHub MCP server ----------------------------------------------------
    github_token: str = ""
    """Fine-grained PAT, 'Public repositories (read-only)' is enough. Empty = GitHub tools disabled."""
    github_mcp_mode: Literal["stdio", "docker", "remote", "http"] = "stdio"
    github_mcp_command: str = "github-mcp-server"
    """Path to the github-mcp-server binary (mode=stdio)."""
    github_mcp_image: str = "ghcr.io/github/github-mcp-server:v2.2.0"
    """Docker image (mode=docker)."""
    github_mcp_url: str = "https://api.githubcopilot.com/mcp/"
    """Remote or self-hosted HTTP endpoint (mode=remote or http)."""
    github_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_GITHUB_TOOLS))

    # --- Codebase MCP server (ours) ------------------------------------------
    codebase_enabled: bool = True
    codebase_cache_dir: str = ".cache/repos"
    codebase_git_base_url: str = "https://github.com"
    codebase_cache_ttl_s: int = 3600
    codebase_max_repo_mb: int = 300
    codebase_clone_timeout_s: int = 120
    """Must stay below TOOL_TIMEOUT_S: the first question about a repository waits for its clone."""

    # --- Agent ----------------------------------------------------------------
    max_tool_calls: int = 8
    """Hard budget per question. When it's spent, the model must answer with what it has."""
    max_tool_output_chars: int = 6000
    """Each tool result is cut to this size before the LLM sees it (small context windows)."""
    tool_timeout_s: float = 150.0
    """Per tool call. MCP sessions also time out a request after this + 30 s, so nothing waits forever."""
    max_history_messages: int = 6

    # --- API --------------------------------------------------------------------
    api_key: str = ""
    """If set, every request needs the header `X-API-Key`. Set it before exposing the API."""
    request_timeout_s: float = 300.0
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()