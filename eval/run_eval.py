"""Measure the agent on a golden set of questions.

    uv run python -m eval.run_eval                      # psf/requests, eval/golden.jsonl
    uv run python -m eval.run_eval --limit 3            # quick check
    uv run python -m eval.run_eval --repo owner/name --golden my_questions.jsonl

Metrics (all deterministic, no LLM judge):
  answer_accuracy     answerable questions whose answer contains every `must_include` fact
  tool_selection      questions where the agent used at least one of the `expected_tools`
  citation_accuracy   answers whose sources include one of the `expected_paths`
  refusal_accuracy    unanswerable questions that were declined (no invented answer)
  avg_tool_calls, latency p50/p95, error_rate

Reports go to eval/results/ (report.md to show, report.json with every question).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from reposcout.agent import RepoAgent
from reposcout.config import get_settings
from reposcout.llm import build_chat_model
from reposcout.mcp_client import McpToolbox
from reposcout.repo_ref import parse_repo

HERE = Path(__file__).parent
REFUSAL_HINTS = (
    "couldn't",
    "could not",
    "can't",
    "cannot",
    "not find",
    "no information",
    "don't know",
    "not related",
    "unable",
    "won't guess",
    "not available",
    "no mention",
    "not mention",
)


def score(item: dict, result: dict) -> dict:
    answer = result.get("answer", "").lower()
    used = set(result.get("tools_used", []))
    labels = " ".join(s["label"] for s in result.get("sources", []))
    answerable = item.get("answerable", True)
    row = {
        "id": item["id"],
        "outcome": result.get("outcome"),
        "tool_calls": result.get("tool_calls", 0),
        "latency_s": result.get("latency_s", 0.0),
        "error": result.get("error"),
    }
    if answerable:
        facts = item.get("must_include", [])
        row["answer_ok"] = result.get("outcome") == "answered" and all(
            f.lower() in answer for f in facts
        )
        expected = item.get("expected_tools", [])
        row["tools_ok"] = bool(used & set(expected)) if expected else None
        paths = item.get("expected_paths", [])
        row["citation_ok"] = any(p in labels for p in paths) if paths else None
    else:
        row["refusal_ok"] = result.get("outcome") == "no_evidence" or any(
            h in answer for h in REFUSAL_HINTS
        )
    return row


def summarize(rows: list[dict]) -> dict:
    def rate(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(values) / len(values), 2) if values else None

    latencies = sorted(r["latency_s"] for r in rows if not r.get("error"))
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None
    return {
        "questions": len(rows),
        "answer_accuracy": rate("answer_ok"),
        "tool_selection": rate("tools_ok"),
        "citation_accuracy": rate("citation_ok"),
        "refusal_accuracy": rate("refusal_ok"),
        "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 1)
        if rows
        else None,
        "latency_p50_s": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_s": round(p95, 1) if p95 is not None else None,
        "error_rate": round(sum(1 for r in rows if r.get("error")) / len(rows), 2)
        if rows
        else None,
    }


def to_markdown(summary: dict, rows: list[dict], meta: dict) -> str:
    lines = [
        f"# Evaluation: {meta['repo']} with {meta['llm']}",
        "",
        f"Run at {meta['time']}, {meta['servers']}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    lines += [
        "",
        "| id | outcome | answer | tools | citation | refusal | tool calls | latency (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    mark = {True: "yes", False: "**no**", None: "-"}
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['outcome'] or r.get('error')} | {mark[r.get('answer_ok')]} | {mark[r.get('tools_ok')]} | "
            f"{mark[r.get('citation_ok')]} | {mark[r.get('refusal_ok')]} | {r['tool_calls']} | {r['latency_s']:.1f} |"
        )
    return "\n".join(lines) + "\n"


async def run(repo: str, golden: Path, limit: int | None) -> dict:
    settings = get_settings()
    items = [
        json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    toolbox = McpToolbox(settings)
    await toolbox.start()
    try:
        has_github = toolbox.has_server("github")
        if not has_github:
            skipped = [i["id"] for i in items if i.get("needs") == "github"]
            print(f"GitHub tools unavailable: skipping {', '.join(skipped)}")
            items = [i for i in items if i.get("needs") != "github"]
        items = items[:limit] if limit else items
        agent = RepoAgent(build_chat_model(settings), toolbox, settings)
        ref = parse_repo(repo)
        details, rows = [], []
        for item in items:
            started = time.perf_counter()
            try:
                res = await agent.ask(ref, item["question"])
                result = {
                    "answer": res.answer,
                    "outcome": res.outcome,
                    "tool_calls": res.tool_calls,
                    "sources": res.sources,
                    "tools_used": [s["tool"] for s in res.steps if s.get("node") == "tool"],
                    "steps": res.steps,
                }
            except Exception as exc:  # noqa: BLE001 - one failure shouldn't stop the run
                result = {"error": f"{type(exc).__name__}: {exc}"[:200], "outcome": None}
            result["latency_s"] = time.perf_counter() - started
            row = score(item, result)
            rows.append(row)
            details.append({**item, **result, "score": row})
            print(
                f"{item['id']:8} {row.get('outcome') or 'error':12} {result['latency_s']:6.1f}s  "
                f"{(result.get('answer') or result.get('error') or '')[:90]!r}"
            )
        summary = summarize(rows)
        servers = ", ".join(f"{n}: {'ok' if s.ok else 'off'}" for n, s in toolbox.status.items())
        meta = {
            "repo": repo,
            "llm": f"{settings.llm_provider}/{settings.llm_model}",
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "servers": servers,
        }
        out = HERE / "results"
        out.mkdir(exist_ok=True)
        (out / "report.json").write_text(
            json.dumps({"meta": meta, "summary": summary, "questions": details}, indent=2),
            encoding="utf-8",
        )
        (out / "report.md").write_text(to_markdown(summary, rows, meta), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Reports: {out / 'report.md'}")
        return summary
    finally:
        await toolbox.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default="psf/requests")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run(args.repo, args.golden, args.limit))


if __name__ == "__main__":
    main()
