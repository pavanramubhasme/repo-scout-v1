"""Streamlit chat UI for RepoScout.

uv run streamlit run ui/app.py
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx
import streamlit as st

API_URL = os.environ.get("REPOSCOUT_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("REPOSCOUT_API_KEY", "")
EXAMPLES = {
    "Code": [
        "What does this project do, and how is it organised?",
        "How are retries configured in HTTPAdapter?",
        "Where is the Session class defined?",
    ],
    "Activity": [
        "What is the latest release?",
        "Summarize the three most recent commits.",
        "Are there open issues about proxies?",
    ],
}

st.set_page_config(page_title="RepoScout", page_icon=":mag:", layout="wide")


def http() -> httpx.Client:
    # A company proxy must never intercept calls to a local API.
    local = urlparse(API_URL).hostname in {"localhost", "127.0.0.1", "::1", "api"}
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    return httpx.Client(base_url=API_URL, timeout=330, trust_env=not local, headers=headers)


@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict | None:
    try:
        with http() as client:
            return client.get("/health").json()
    except Exception:  # noqa: BLE001
        return None


def reset() -> None:
    st.session_state.messages = []


def render_details(meta: dict) -> None:
    if meta.get("sources"):
        st.markdown(
            "**Sources:** " + " · ".join(f"[{s['label']}]({s['url']})" for s in meta["sources"])
        )
    steps = meta.get("steps", [])
    title = f"How it was answered: {meta.get('tool_calls', 0)} tool calls, {meta.get('latency_ms', 0) / 1000:.1f} s"
    if meta.get("outcome") != "answered":
        title += f" ({meta.get('outcome')})"
    with st.expander(title):
        for step in steps:
            if step["node"] == "agent":
                nxt = ", ".join(step.get("tool_calls", [])) or "final answer"
                st.markdown(f"🧠 **LLM** → {nxt} · {step.get('ms', 0)} ms")
                if step.get("recovered_text_calls"):
                    st.caption(
                        "policy: tool call written as text by the model, recovered and executed"
                    )
            elif step["node"] == "tool":
                icon = "✅" if step.get("ok") else "⚠️"
                args = ", ".join(f"{k}={v!r}" for k, v in (step.get("args") or {}).items())
                extra = " · truncated" if step.get("truncated") else ""
                st.markdown(
                    f"{icon} `{step['tool']}({args})` · {step.get('server', '?')} · {step.get('ms', 0)} ms{extra}"
                )
                for note in step.get("notes", []):
                    st.caption(f"policy: {note}")
                if step.get("blocked"):
                    st.caption(f"blocked: {step['blocked']}")
                elif step.get("duplicate"):
                    st.caption("duplicate call, not executed")
                elif step.get("error"):
                    st.caption(f"error: {step['error'][:300]}")
            elif step["node"] == "verify":
                st.markdown(f"🔎 **verify** → {step.get('result')}")


if "messages" not in st.session_state:
    reset()

with st.sidebar:
    st.title("🔎 RepoScout")
    st.caption("Ask questions about any public GitHub repository.")
    repo = st.text_input(
        "Repository",
        value=st.session_state.get("repo", "psf/requests"),
        help="owner/name or a github.com URL",
    )
    if repo != st.session_state.get("repo"):
        st.session_state.repo = repo
        reset()
    if st.button("New conversation", width="stretch"):
        reset()

    status = health()
    st.divider()
    if status is None:
        st.error(f"API not reachable at {API_URL}")
    else:
        st.caption(f"LLM: {status['llm']['provider']} / {status['llm']['model']}")
        for name, server in status["servers"].items():
            if server["ok"]:
                st.caption(f"🟢 {name}: {len(server['tools'])} tools")
            else:
                st.caption(f"⚪ {name}: {server.get('error') or 'unavailable'}")

    st.divider()
    clicked = None
    for group, questions in EXAMPLES.items():
        st.caption(group)
        for question in questions:
            if st.button(question, key=question, width="stretch"):
                clicked = question

st.subheader(f"Ask about `{st.session_state.repo}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            render_details(message["meta"])

question = st.chat_input("e.g. How is authentication implemented?") or clicked
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][-6:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Looking through the repository..."):
            try:
                with http() as client:
                    response = client.post(
                        "/ask",
                        json={
                            "repo": st.session_state.repo,
                            "question": question,
                            "history": history,
                        },
                    )
                if response.status_code != 200:
                    try:
                        detail = response.json().get("detail")
                    except ValueError:
                        detail = response.text[:300]
                    st.error(f"{response.status_code}: {detail}")
                    st.session_state.messages.pop()
                    st.stop()
                data = response.json()
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the API at {API_URL}: {exc}")
                st.session_state.messages.pop()
                st.stop()
        st.markdown(data["answer"])
        render_details(data)
    st.session_state.messages.append({"role": "assistant", "content": data["answer"], "meta": data})