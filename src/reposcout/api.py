"""REST API.

uv run uvicorn reposcout.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .agent import RepoAgent
from .config import Settings, get_settings
from .llm import build_chat_model
from .mcp_client import McpToolbox, redact
from .repo_ref import InvalidRepo, parse_repo

log = logging.getLogger("reposcout")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    repo: str = Field(examples=["psf/requests"], max_length=200)
    question: str = Field(
        min_length=1, max_length=2000, examples=["How are retries configured in HTTPAdapter?"]
    )
    history: list[Turn] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    label: str
    url: str
    kind: str


class AskResponse(BaseModel):
    request_id: str
    repo: str
    answer: str
    outcome: str
    sources: list[Source]
    tool_calls: int
    latency_ms: int
    steps: list[dict]


def _is_connection_error(exc: BaseException) -> bool:
    names = {type(e).__name__ for e in (exc, exc.__cause__, exc.__context__) if e is not None}
    return bool(
        names
        & {
            "ConnectError",
            "ConnectionError",
            "APIConnectionError",
            "ConnectTimeout",
            "ResponseError",
        }
    )


def create_app(settings: Settings | None = None, agent: RepoAgent | None = None) -> FastAPI:
    """`agent` is injectable so the tests can run the API without real servers or an LLM."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        toolbox = None
        if agent is None:
            toolbox = McpToolbox(settings)
            await toolbox.start()
            app.state.agent = RepoAgent(build_chat_model(settings), toolbox, settings)
        else:
            app.state.agent = agent
        try:
            yield
        finally:
            if toolbox is not None:
                await toolbox.aclose()

    app = FastAPI(
        title="RepoScout",
        version=__version__,
        lifespan=lifespan,
        description="Ask questions about any public GitHub repository.",
    )

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
            raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")

    @app.get("/health")
    async def health(request: Request) -> dict:
        repo_agent: RepoAgent = request.app.state.agent
        servers = {name: vars(status) for name, status in repo_agent.toolbox.status.items()}
        ok = bool(repo_agent.toolbox.specs)
        degraded = any(not s.ok for s in repo_agent.toolbox.status.values())
        return {
            "status": "ok" if ok and not degraded else ("degraded" if ok else "no_tools"),
            "version": __version__,
            "llm": {"provider": settings.llm_provider, "model": settings.llm_model},
            "servers": servers,
        }

    @app.get("/tools", dependencies=[Depends(require_key)])
    async def tools(request: Request) -> list[dict]:
        repo_agent: RepoAgent = request.app.state.agent
        return [
            {
                "name": s.name,
                "server": s.server,
                "description": s.tool.description,
                "parameters": s.llm_schema,
            }
            for s in repo_agent.toolbox.specs.values()
        ]

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        request_id = uuid.uuid4().hex[:12]
        try:
            repo = parse_repo(body.repo)
        except InvalidRepo as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        repo_agent: RepoAgent = request.app.state.agent
        if not repo_agent.toolbox.specs:
            raise HTTPException(
                status_code=503, detail="No MCP tools are available. Check /health."
            )
        history = [t.model_dump() for t in body.history]
        try:
            result = await asyncio.wait_for(
                repo_agent.ask(repo, body.question, history), timeout=settings.request_timeout_s
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504, detail="The agent took too long. Try a narrower question."
            ) from exc
        except Exception as exc:
            log.exception("ask_failed request_id=%s", request_id)
            if _is_connection_error(exc):
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM backend unavailable ({settings.llm_provider} at {settings.llm_base_url}).",
                ) from exc
            raise HTTPException(
                status_code=500, detail=redact(f"{type(exc).__name__}: {exc}", settings)[:300]
            ) from exc
        log.info(
            "ask request_id=%s repo=%s outcome=%s tool_calls=%d latency_ms=%d",
            request_id,
            repo,
            result.outcome,
            result.tool_calls,
            result.latency_ms,
        )
        return AskResponse(
            request_id=request_id,
            repo=repo.full_name,
            answer=result.answer,
            outcome=result.outcome,
            sources=[Source(**s) for s in result.sources],
            tool_calls=result.tool_calls,
            latency_ms=result.latency_ms,
            steps=result.steps,
        )

    return app


app = create_app()
