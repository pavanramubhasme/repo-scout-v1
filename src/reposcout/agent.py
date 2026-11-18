"""The agent: a small LangGraph state machine around an LLM with MCP tools.

    START -> agent --(tool calls)--> tools --> agent ... --(no tool calls)--> verify --> END
                                                                                 +--(nudge)--> agent

* agent  : the LLM decides the next step. When the tool budget is spent, it gets no tools
           and must answer with what it has.
* tools  : runs the requested MCP tools in parallel, through the policy layer (scope,
           repair, defaults), with timeouts, de-duplication and output truncation.
* verify : deterministic evidence check. An answer with no successful tool call behind
           it gets one nudge ("check first"); if there is still no evidence, the agent
           refuses instead of answering from memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import prompts
from .citations import dedupe, sources_for_call
from .codebase.core import redact_secrets
from .config import Settings
from .mcp_client import McpToolbox, redact
from .repo_ref import RepoRef
from .tool_policy import PolicyViolation, call_signature, prepare_args

log = logging.getLogger(__name__)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

Outcome = Literal["answered", "no_evidence", "error"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    repo: str
    tool_calls_used: int
    evidence: int
    nudged: bool
    outcome: str
    seen_calls: dict[str, str]
    sources: Annotated[list[dict[str, str]], operator.add]
    steps: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class AgentResult:
    answer: str
    outcome: str
    sources: list[dict[str, str]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    latency_ms: int = 0


def content_to_text(content: Any) -> str:
    """MCP results arrive as a string or a list of content blocks; the LLM needs plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, dict):
            parts.append(f"[{block.get('type', 'content')} omitted]")
    return "\n".join(parts)


_TEXT_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(?P<tag>\{.*?\})\s*</tool_call>|```(?:json|tool_call)?\s*(?P<fence>\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


_NAME_KEYS = ("name", "tool", "tool_name", "function", "action")
_ARG_KEYS = ("arguments", "parameters", "args", "input", "tool_input", "action_input")


def _as_call(blob: str, known: set[str]) -> dict[str, Any] | None:
    """Accept the shapes small models invent: {"name", "arguments"}, {"tool", "parameters"},
    {"function": {...}}, or the arguments flattened next to the tool name."""
    try:
        data = json.loads(blob)
    except ValueError:
        return None
    if isinstance(data, dict) and isinstance(data.get("function"), dict):
        data = data["function"]
    if not isinstance(data, dict):
        return None
    name_key = next(
        (k for k in _NAME_KEYS if isinstance(data.get(k), str) and data[k] in known), None
    )
    if name_key is None:
        return None
    arg_key = next((k for k in _ARG_KEYS if k in data), None)
    args = data[arg_key] if arg_key else {k: v for k, v in data.items() if k != name_key}
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {
        "name": data[name_key],
        "args": args,
        "id": f"text_{uuid.uuid4().hex[:8]}",
        "type": "tool_call",
    }


def extract_text_tool_calls(text: str, known: set[str]) -> tuple[list[dict[str, Any]], str]:
    """Small local models sometimes *write* a tool call as JSON in their reply (in a ```json
    block or <tool_call> tags) instead of making a structured call. Return those calls and
    the text without them. Only names of real tools count, so ordinary JSON stays text."""
    calls: list[dict[str, Any]] = []

    def take(match: re.Match) -> str:
        call = _as_call(match.group("tag") or match.group("fence"), known)
        if call is None:
            return match.group(0)
        calls.append(call)
        return ""

    rest = _TEXT_CALL_BLOCK.sub(take, text or "")
    stripped = rest.strip()
    if not calls and stripped.startswith("{") and stripped.endswith("}"):
        call = _as_call(stripped, known)
        if call:
            return [call], ""
    return calls, rest.strip()


def clean_answer(text: str) -> str:
    return redact_secrets(_THINK.sub("", text or "")).strip()


def truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return (
        text[:limit]
        + f"\n[... truncated {len(text) - limit:,} characters. Ask for less: a narrower path, query or line range.]",
        True,
    )


class RepoAgent:
    def __init__(self, llm: BaseChatModel, toolbox: McpToolbox, settings: Settings):
        self.llm = llm
        self.toolbox = toolbox
        self.settings = settings
        tools = toolbox.llm_tools()
        self.llm_with_tools = llm.bind_tools(tools) if tools else llm
        self.graph = self._build_graph()

    # --- graph -----------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent)
        graph.add_node("tools", self._tools)
        graph.add_node("verify", self._verify)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", self._after_agent, {"tools": "tools", "verify": "verify"}
        )
        graph.add_edge("tools", "agent")
        graph.add_conditional_edges("verify", self._after_verify, {"agent": "agent", "end": END})
        return graph.compile()

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def _budget_left(self, state: AgentState) -> int:
        return self.settings.max_tool_calls - state.get("tool_calls_used", 0)

    def _context_view(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Keep the prompt inside a small context window: the newest tool results stay whole,
        older ones are shortened (the state itself keeps everything)."""
        budget = self.settings.max_tool_output_chars * 4
        view: list[AnyMessage] = []
        used = 0
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                text = content_to_text(message.content)
                if used + len(text) > budget:
                    text = text[:400] + "\n[... earlier result shortened to save context]"
                    message = message.model_copy(update={"content": text})
                used += len(text)
            view.append(message)
        return list(reversed(view))

    async def _agent(self, state: AgentState) -> dict[str, Any]:
        budget = self._budget_left(state)
        system = SystemMessage(
            prompts.system_prompt(
                state["repo"],
                budget,
                self.toolbox.has_server("codebase"),
                self.toolbox.has_server("github"),
            )
        )
        messages: list[AnyMessage] = [system, *self._context_view(state["messages"])]
        model = self.llm_with_tools
        if budget <= 0:
            model = self.llm  # no tools bound: the model has to answer now
            messages.append(HumanMessage(prompts.BUDGET_EXHAUSTED))
        started = time.perf_counter()
        response = await model.ainvoke(messages)
        ms = int((time.perf_counter() - started) * 1000)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=content_to_text(response.content))
        recovered = 0
        if not response.tool_calls:
            text_calls, rest = extract_text_tool_calls(
                content_to_text(response.content), set(self.toolbox.specs)
            )
            if text_calls:
                # Executed like real calls when budget is left; never shown to the user as an answer.
                recovered = len(text_calls) if budget > 0 else 0
                # With budget left, the narration around the call is dropped: it's a plan, not an
                # answer, and small models put invented "tool results" in it.
                response = AIMessage(
                    content="" if budget > 0 else rest,
                    tool_calls=text_calls if budget > 0 else [],
                    id=response.id,
                )
        calls = list(response.tool_calls or [])
        if budget <= 0 and calls:
            calls = []
        if len(calls) > max(budget, 0):
            calls = calls[: max(budget, 0)]
        if calls != list(response.tool_calls or []):
            response = AIMessage(content=response.content, tool_calls=calls, id=response.id)
        step = {"node": "agent", "ms": ms, "tool_calls": [c["name"] for c in calls]}
        if recovered:
            step["recovered_text_calls"] = recovered
        return {"messages": [response], "steps": [step]}

    def _after_agent(self, state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "verify"

    async def _tools(self, state: AgentState) -> dict[str, Any]:
        repo = RepoRef(*state["repo"].split("/", 1))
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        seen = dict(state.get("seen_calls") or {})
        results = await asyncio.gather(
            *(self._run_one(call, repo, seen) for call in last.tool_calls)
        )
        messages, sources, steps = [], [], []
        evidence = 0
        for message, call_sources, step, ok, signature in results:
            messages.append(message)
            sources.extend(call_sources)
            steps.append(step)
            evidence += int(ok)
            if signature:
                seen[signature] = message.tool_call_id
        return {
            "messages": messages,
            "sources": sources,
            "steps": steps,
            "evidence": state.get("evidence", 0) + evidence,
            "tool_calls_used": state.get("tool_calls_used", 0) + len(last.tool_calls),
            "seen_calls": seen,
        }

    async def _run_one(self, call: dict[str, Any], repo: RepoRef, seen: dict[str, str]):
        name, call_id = call["name"], call["id"]
        step: dict[str, Any] = {"node": "tool", "tool": name, "ok": False}

        def reply(text: str, ok: bool, signature: str | None = None, sources=None):
            text, truncated = truncate(text, self.settings.max_tool_output_chars)
            step.update(ok=ok, chars=len(text), truncated=truncated)
            if not ok:
                step["error"] = text[
                    :300
                ]  # visible in the trace: the first thing to read when debugging
            status = "success" if ok else "error"
            wrapped = f'<tool_result tool="{name}" status="{status}">\n{text}\n</tool_result>'
            return (
                ToolMessage(content=wrapped, tool_call_id=call_id, name=name, status=status),
                sources or [],
                step,
                ok,
                signature,
            )

        spec = self.toolbox.specs.get(name)
        if spec is None:
            available = ", ".join(sorted(self.toolbox.specs)) or "none"
            return reply(f"Unknown tool {name!r}. Available tools: {available}.", False)
        step["server"] = spec.server
        try:
            args, notes = prepare_args(spec, call.get("args"), repo)
        except PolicyViolation as exc:
            step["blocked"] = str(exc)
            return reply(f"Call refused: {exc}", False)
        step["args"] = {k: v for k, v in args.items() if k not in ("owner", "repo", "fields")}
        if notes:
            step["notes"] = notes
        signature = call_signature(name, args)
        if signature in seen:
            step["duplicate"] = True
            return reply(
                "You already made this exact call; its result is above. Use it, or try something different.",
                False,
            )

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                spec.tool.ainvoke({"type": "tool_call", "name": name, "args": args, "id": call_id}),
                timeout=self.settings.tool_timeout_s,
            )
            text = content_to_text(result.content if isinstance(result, ToolMessage) else result)
            ok = not (isinstance(result, ToolMessage) and result.status == "error")
        except TimeoutError:
            text, ok = f"The tool timed out after {self.settings.tool_timeout_s:.0f} s.", False
        except Exception as exc:  # noqa: BLE001 - every failure becomes a message the model can react to
            text, ok = f"Tool failed: {type(exc).__name__}: {exc}", False
        step["ms"] = int((time.perf_counter() - started) * 1000)
        text = redact_secrets(redact(text, self.settings))
        if ok and not text.strip():
            text, ok = "(empty result)", False
        sources = sources_for_call(name, args, repo, text) if ok else []
        return reply(text, ok, signature if ok else None, sources)

    def _verify(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        answer = clean_answer(content_to_text(last.content)) if isinstance(last, AIMessage) else ""
        has_evidence = state.get("evidence", 0) > 0
        can_retry = not state.get("nudged") and self._budget_left(state) > 0
        if has_evidence and answer:
            return {"outcome": "answered", "steps": [{"node": "verify", "result": "answered"}]}
        if can_retry:
            nudge = prompts.NUDGE if not has_evidence else prompts.EMPTY_NUDGE
            return {
                "messages": [HumanMessage(nudge)],
                "nudged": True,
                "steps": [{"node": "verify", "result": "nudge"}],
            }
        if has_evidence and not answer:
            text = "I found information in the repository but couldn't put an answer together. Please rephrase the question."
            return {
                "messages": [AIMessage(text)],
                "outcome": "error",
                "steps": [{"node": "verify", "result": "empty_answer"}],
            }
        return {
            "messages": [AIMessage(prompts.NO_EVIDENCE)],
            "outcome": "no_evidence",
            "steps": [{"node": "verify", "result": "no_evidence"}],
        }

    def _after_verify(self, state: AgentState) -> str:
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else "end"

    # --- public API ------------------------------------------------------------

    async def ask(
        self, repo: RepoRef, question: str, history: list[dict[str, str]] | None = None
    ) -> AgentResult:
        started = time.perf_counter()
        messages: list[AnyMessage] = []
        for turn in (history or [])[-self.settings.max_history_messages :]:
            content = str(turn.get("content", ""))[:4000]
            if turn.get("role") == "user":
                messages.append(HumanMessage(content))
            elif turn.get("role") == "assistant":
                messages.append(AIMessage(content))
        messages.append(HumanMessage(question))
        state: AgentState = {
            "messages": messages,
            "repo": repo.full_name,
            "tool_calls_used": 0,
            "evidence": 0,
            "nudged": False,
            "seen_calls": {},
            "sources": [],
            "steps": [],
        }
        limit = 4 * self.settings.max_tool_calls + 10
        final = await self.graph.ainvoke(state, config={"recursion_limit": limit})
        last = final["messages"][-1]
        answer = clean_answer(content_to_text(last.content))
        result = AgentResult(
            answer=answer,
            outcome=final.get("outcome", "answered"),
            sources=dedupe(final.get("sources", [])),
            steps=final.get("steps", []),
            tool_calls=final.get("tool_calls_used", 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        log.info(
            "question_answered repo=%s outcome=%s tool_calls=%d latency_ms=%d",
            repo.full_name,
            result.outcome,
            result.tool_calls,
            result.latency_ms,
        )
        return result