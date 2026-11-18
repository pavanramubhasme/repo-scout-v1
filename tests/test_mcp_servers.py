"""Integration: real stdio subprocesses, exactly as the API starts them."""

import os
import shutil

import pytest

from reposcout.config import DEFAULT_GITHUB_TOOLS
from reposcout.mcp_client import McpToolbox, codebase_connection, github_connection
from reposcout.repo_ref import parse_repo
from reposcout.tool_policy import prepare_args
from tests.conftest import make_settings


async def test_codebase_server_over_stdio(tmp_path, git_remote):
    settings = make_settings(
        codebase_cache_dir=str(tmp_path / "cache"),
        codebase_git_base_url=git_remote,
        github_token="",
    )
    box = McpToolbox(settings)
    await box.start()
    try:
        assert box.status["codebase"].ok, box.status["codebase"].error
        assert set(box.specs) == {"repo_overview", "list_files", "grep_code", "read_file"}
        assert box.status["github"].error == "set GITHUB_TOKEN to enable"
        spec = box.specs["grep_code"]
        args, _ = prepare_args(spec, {"query": "MAX_WIDGETS"}, parse_repo("acme/widgets"))
        result = await spec.tool.ainvoke(
            {"type": "tool_call", "name": "grep_code", "args": args, "id": "1"}
        )
        assert "src/widgets/core.py:1: MAX_WIDGETS = 42" in str(result.content)
    finally:
        await box.aclose()


async def test_broken_server_does_not_stop_the_others(tmp_path, git_remote):
    settings = make_settings(
        codebase_cache_dir=str(tmp_path / "cache"),
        codebase_git_base_url=git_remote,
        github_token="ghp_x",
        github_mcp_command=str(tmp_path / "missing-binary"),
    )
    box = McpToolbox(settings)
    await box.start()
    try:
        assert box.status["github"].ok is False and "ghp_x" not in (
            box.status["github"].error or ""
        )
        assert box.status["codebase"].ok is True
    finally:
        await box.aclose()


def test_github_connection_modes():
    base = {"github_token": "ghp_test", "github_tools": ["list_issues", "issue_read"]}
    stdio = github_connection(make_settings(**base))
    assert stdio["command"] == "github-mcp-server" and stdio["args"] == ["stdio"]
    assert (
        stdio["env"]["GITHUB_READ_ONLY"] == "1"
        and stdio["env"]["GITHUB_TOOLS"] == "list_issues,issue_read"
    )
    docker = github_connection(make_settings(github_mcp_mode="docker", **base))
    assert docker["command"] == "docker" and "ghp_test" not in " ".join(
        docker["args"]
    )  # token via env, not argv
    remote = github_connection(make_settings(github_mcp_mode="remote", **base))
    assert remote["transport"] == "streamable_http"
    assert (
        remote["headers"]["X-MCP-Readonly"] == "true"
        and remote["headers"]["Authorization"] == "Bearer ghp_test"
    )
    for conn in (stdio, docker, remote, codebase_connection(make_settings(tool_timeout_s=60))):
        # a stuck server or hung subprocess must end in an error, never an endless wait
        assert conn["session_kwargs"]["read_timeout_seconds"].total_seconds() >= 90
    assert github_connection(make_settings(github_token="")) is None
    assert codebase_connection(make_settings(codebase_enabled=False)) is None


GITHUB_BINARY = os.environ.get("GITHUB_MCP_BINARY") or shutil.which("github-mcp-server")


@pytest.mark.skipif(not GITHUB_BINARY, reason="github-mcp-server binary not installed")
async def test_real_github_mcp_server_registers_only_read_only_allowlisted_tools(tmp_path):
    # A dummy token is enough to start the server and list its tools (no API call is made).
    settings = make_settings(
        github_token="ghp_dummy_for_listing_only",
        github_mcp_command=GITHUB_BINARY,
        codebase_enabled=False,
        github_tools=[*DEFAULT_GITHUB_TOOLS],
    )
    box = McpToolbox(settings)
    await box.start()
    try:
        assert box.status["github"].ok, box.status["github"].error
        assert set(box.specs) == set(DEFAULT_GITHUB_TOOLS)
        for spec in box.specs.values():
            assert (
                "owner" not in spec.llm_schema["properties"]
                and "repo" not in spec.llm_schema["properties"]
            )
        assert box.specs["list_issues"].schema["properties"]["state"]["enum"] == ["OPEN", "CLOSED"]
    finally:
        await box.aclose()