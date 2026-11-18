from pathlib import Path

from eval.run_eval import score, summarize
from reposcout.llm import build_chat_model, is_local, resolve_base_url
from reposcout.mcp_client import resolve_command
from tests.conftest import make_settings


def test_docker_rewrites_loopback_llm_url(monkeypatch):
    assert resolve_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    monkeypatch.setenv("REPOSCOUT_IN_DOCKER", "1")
    assert resolve_base_url("http://127.0.0.1:11434") == "http://host.docker.internal:11434"
    assert resolve_base_url("https://api.groq.com/openai/v1") == "https://api.groq.com/openai/v1"


def test_local_llm_bypasses_proxies():
    assert is_local("http://127.0.0.1:11434") and not is_local("https://api.groq.com/openai/v1")
    model = build_chat_model(make_settings())
    assert model.client_kwargs["trust_env"] is False and model.num_ctx == 16384


def test_openai_compatible_provider():
    model = build_chat_model(
        make_settings(
            llm_provider="openai",
            llm_model="openai/gpt-oss-20b",
            llm_base_url="https://api.groq.com/openai/v1",
            llm_api_key="gsk_x",
        )
    )
    assert model.model_name == "openai/gpt-oss-20b" and model.temperature == 0


def test_relative_command_paths_are_resolved():
    assert resolve_command("github-mcp-server") == "github-mcp-server"  # found on PATH
    assert resolve_command("tools/github-mcp-server.exe") == str(
        Path.cwd() / "tools" / "github-mcp-server.exe"
    )


def test_eval_scoring():
    item = {
        "id": "a",
        "question": "q",
        "must_include": ["10"],
        "expected_tools": ["grep_code"],
        "expected_paths": ["adapters.py"],
    }
    good = score(
        item,
        {
            "answer": "Pool size is 10",
            "outcome": "answered",
            "tools_used": ["grep_code"],
            "sources": [{"label": "src/adapters.py:L80"}],
            "latency_s": 1.0,
            "tool_calls": 1,
        },
    )
    assert good["answer_ok"] and good["tools_ok"] and good["citation_ok"]
    refusal = score(
        {"id": "n", "question": "q", "answerable": False},
        {
            "answer": "I couldn't find that.",
            "outcome": "answered",
            "latency_s": 2.0,
            "tool_calls": 1,
        },
    )
    assert refusal["refusal_ok"]
    summary = summarize([good, refusal])
    assert (
        summary["answer_accuracy"] == 1.0
        and summary["refusal_accuracy"] == 1.0
        and summary["latency_p50_s"] == 1.5
    )
