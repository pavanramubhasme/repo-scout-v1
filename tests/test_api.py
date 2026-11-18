"""REST API with a stub agent (no LLM, no MCP servers)."""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from reposcout.agent import AgentResult
from reposcout.api import create_app
from reposcout.mcp_client import ServerStatus
from tests.conftest import make_settings


class StubToolbox:
    def __init__(self, ok=True):
        self.specs = {"grep_code": object()} if ok else {}
        self.status = {
            "codebase": ServerStatus(
                "codebase", ok=ok, transport="stdio", tools=["grep_code"] if ok else []
            ),
            "github": ServerStatus("github", ok=False, error="set GITHUB_TOKEN to enable"),
        }


class StubAgent:
    def __init__(self, behaviour="answer", ok=True):
        self.toolbox = StubToolbox(ok)
        self.behaviour = behaviour
        self.calls = []

    async def ask(self, repo, question, history):
        self.calls.append((repo.full_name, question, history))
        if self.behaviour == "slow":
            await asyncio.sleep(5)
        if self.behaviour == "llm_down":
            raise httpx.ConnectError("connection refused")
        if self.behaviour == "boom":
            raise RuntimeError("bad thing with token ghp_supersecret")
        return AgentResult(
            answer="It's 42.",
            outcome="answered",
            tool_calls=2,
            latency_ms=12,
            sources=[
                {
                    "label": "src/a.py:L1",
                    "url": "https://github.com/a/b/blob/HEAD/src/a.py#L1",
                    "kind": "code",
                }
            ],
            steps=[{"node": "agent"}, {"node": "tool", "tool": "grep_code", "ok": True}],
        )


@pytest.fixture
def client_factory():
    def make(agent=None, **settings):
        app = create_app(make_settings(**settings), agent=agent or StubAgent())
        return TestClient(app)

    return make


def test_ask_returns_answer_sources_and_trace(client_factory):
    agent = StubAgent()
    with client_factory(agent) as client:
        response = client.post(
            "/ask",
            json={
                "repo": "https://github.com/psf/requests",
                "question": "Limit?",
                "history": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert (
        body["repo"] == "psf/requests"
        and body["answer"] == "It's 42."
        and body["outcome"] == "answered"
    )
    assert body["sources"][0]["url"].endswith("#L1") and body["tool_calls"] == 2
    assert len(body["request_id"]) == 12
    assert agent.calls[0][2] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"repo": "not a repo", "question": "q"},
        {"repo": "psf/requests", "question": ""},
        {"repo": "psf/requests", "question": "x" * 2001},
        {"repo": "psf/requests", "question": "q", "history": [{"role": "system", "content": "x"}]},
    ],
)
def test_invalid_requests(client_factory, payload):
    with client_factory() as client:
        assert client.post("/ask", json=payload).status_code == 422


def test_api_key(client_factory):
    with client_factory(api_key="s3cret") as client:
        assert client.post("/ask", json={"repo": "a/b", "question": "q"}).status_code == 401
        assert (
            client.post(
                "/ask", json={"repo": "a/b", "question": "q"}, headers={"X-API-Key": "wrong"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/ask", json={"repo": "a/b", "question": "q"}, headers={"X-API-Key": "s3cret"}
            ).status_code
            == 200
        )
        assert client.get("/health").status_code == 200  # health stays open for load balancers


def test_health_reports_degraded_servers(client_factory):
    with client_factory() as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["servers"]["codebase"]["ok"] is True
    assert body["servers"]["github"]["error"] == "set GITHUB_TOKEN to enable"


def test_no_tools_means_503(client_factory):
    with client_factory(StubAgent(ok=False)) as client:
        assert client.get("/health").json()["status"] == "no_tools"
        assert client.post("/ask", json={"repo": "a/b", "question": "q"}).status_code == 503


def test_timeout_and_llm_down(client_factory):
    with client_factory(StubAgent("slow"), request_timeout_s=0.1) as client:
        assert client.post("/ask", json={"repo": "a/b", "question": "q"}).status_code == 504
    with client_factory(StubAgent("llm_down")) as client:
        response = client.post("/ask", json={"repo": "a/b", "question": "q"})
        assert (
            response.status_code == 503 and "LLM backend unavailable" in response.json()["detail"]
        )


def test_internal_errors_do_not_leak_secrets(client_factory):
    with client_factory(StubAgent("boom"), github_token="ghp_supersecret") as client:
        response = client.post("/ask", json={"repo": "a/b", "question": "q"})
    assert response.status_code == 500
    assert "ghp_supersecret" not in response.text and "***" in response.json()["detail"]
