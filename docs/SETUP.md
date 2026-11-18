# Setup and run, step by step (Windows PowerShell, uv)

macOS/Linux variants are in brackets. Steps 1–8 get the agent answering; 9–13 add the UI, evaluation, Docker and cloud deployment.

## Step 1 — Install the tools (one time)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # uv
winget install Git.Git                                                                # git (the codebase server clones with it)
```

Close and reopen the terminal, then check `uv --version` and `git --version`. **Ollama** from ollama.com/download if you want a local model. **Docker Desktop** is optional.

## Step 2 — Unzip and install

Unzip to a short path such as `C:\Projects\repo-scout` (not OneDrive/Desktop: sync locks files), then:

```powershell
cd C:\Projects\repo-scout
uv sync
```

This creates `.venv`, downloads Python 3.12 if needed and installs the exact versions from `uv.lock`. No activation needed: run everything with `uv run ...`.

## Step 3 — Choose the LLM

**Option A — local Ollama (free, private, slower):**

```powershell
ollama pull granite4:3b          # about 2 GB; you may already have it
```

A 3B model can call tools, but it picks the wrong tool more often and multi-step questions take a while on a CPU. If you have the disk space, `qwen3:8b` (about 5 GB) is noticeably better at tool use: set `LLM_MODEL=qwen3:8b`.

**Option B — Groq (open-weight models, fast, needs an API key from console.groq.com):** in `.env` set

```
LLM_PROVIDER=openai
LLM_MODEL=openai/gpt-oss-20b
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
```

For a live interview demo, Option B is the safer choice (seconds instead of minutes, reliable tool calls); keep Option A to show it also runs fully locally.

## Step 4 — Create a GitHub token (read-only, public repositories)

github.com → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**. Under **Repository access** choose **Public repositories** (read-only). No extra permissions are needed. Copy the token (`github_pat_...`).

Why so narrow: the agent reads untrusted text (issues, READMEs). Even if a prompt injection fooled the model, the token can't read your private repositories or write anything.

## Step 5 — Get the GitHub MCP server

**Option A — the binary (no Docker):**

```powershell
mkdir tools
Invoke-WebRequest https://github.com/github/github-mcp-server/releases/download/v2.2.0/github-mcp-server_Windows_x86_64.zip -OutFile tools\gms.zip
Expand-Archive tools\gms.zip -DestinationPath tools -Force
tools\github-mcp-server.exe --version          # Version: 2.2.0
```

[macOS/Linux: download `github-mcp-server_Darwin_arm64.tar.gz` / `..._Linux_x86_64.tar.gz`, extract to `tools/`, set `GITHUB_MCP_COMMAND=tools/github-mcp-server`]

**Option B — Docker:** Docker Desktop running, then set `GITHUB_MCP_MODE=docker` in `.env` (the image `ghcr.io/github/github-mcp-server:v2.2.0` is pulled on first use).

## Step 6 — Create your config

```powershell
copy .env.example .env        # [cp .env.example .env]
notepad .env
```

Set `GITHUB_TOKEN=github_pat_...` and the LLM lines from Step 3. `GITHUB_MCP_COMMAND=tools/github-mcp-server.exe` is already the default. The app reads `.env` itself; nothing else to configure.

## Step 7 — Run the tests

```powershell
uv run pytest
```

Expect `94 passed, 1 skipped`. The skipped test starts the real GitHub MCP server; to include it:

```powershell
$env:GITHUB_MCP_BINARY = "tools\github-mcp-server.exe"; uv run pytest    # 95 passed
```

The tests need no LLM, no token and no network: they use a scripted LLM, a local git repository and in-memory MCP servers.

## Step 8 — Ask from the command line

```powershell
uv run repo-scout tools
uv run repo-scout ask psf/requests "How are retries configured in HTTPAdapter?"
uv run repo-scout ask psf/requests "What is the latest release?"
uv run repo-scout ask psf/requests "Who won the 2022 FIFA World Cup?"
uv run repo-scout graph
```

`tools` should show `codebase: ok` (4 tools) and `github: ok` (9 tools). Each answer prints its sources and the `path`, e.g. `agent -> grep_code -> agent -> read_file -> agent -> verify`. The off-topic question should end with `outcome: no_evidence` or an answer saying it couldn't find it. `graph` prints the LangGraph workflow as Mermaid (paste into mermaid.live).

The first question about a repository clones it (a few seconds, cached for an hour in `.cache\repos`).

**The agent is working.**

## Step 9 — API and UI (two terminals)

```powershell
uv run uvicorn reposcout.api:app --host 127.0.0.1 --port 8000     # terminal 1
uv run streamlit run ui/app.py                                    # terminal 2
```

API docs at http://127.0.0.1:8000/docs, UI at http://localhost:8501. The sidebar shows which MCP servers are up. Each answer shows its sources and an expander **How it was answered** with every LLM step and tool call, including what the policy layer changed (`repaired state: 'open' -> 'OPEN'`, `ignored model-chosen owner='...'`) or blocked.

## Step 10 — Evaluate (before the interview)

```powershell
uv run python -m eval.run_eval --limit 3      # quick check
uv run python -m eval.run_eval                # 13 questions on psf/requests
Get-Content eval\results\report.md
```

Metrics are deterministic (no LLM judge): answer accuracy (the key fact is in the answer), tool selection, citation accuracy, refusal accuracy for unanswerable questions, tool calls, latency p50/p95. Run it with both LLM options and keep the two reports: that comparison is a good interview talking point.

## Step 11 — Everything in Docker

```powershell
docker compose up --build
```

UI http://localhost:8501, API http://localhost:8000/docs. It uses your `.env`. Ollama stays on the host: `127.0.0.1` in `LLM_BASE_URL` is rewritten to `host.docker.internal` inside the container. The GitHub MCP server binary is built into the API image, so no extra container is needed.

## Step 12 — Deploy to the cloud

Any container platform works (Render, Railway, Fly.io, Azure Container Apps, Google Cloud Run):

1. Push the project to GitHub (CI runs lint, tests and both Docker builds).
2. Create a service from the `Dockerfile` with target `api`, port 8000, health check path `/health`.
3. Set the environment variables: `GITHUB_TOKEN`, the Groq lines from Step 3 (a hosted LLM; a cloud container has no Ollama), and **`API_KEY`** (any long random string: without it anyone could spend your tokens).
4. Optional UI: a second service with target `ui`, port 8501, `REPOSCOUT_API_URL=https://<your-api-url>` and `REPOSCOUT_API_KEY=<same key>`.

## Step 13 — Stop

`Ctrl+C` in both terminals, or `docker compose down`.

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` (any OpenAI-compatible API) |
| `LLM_MODEL` | `granite4:3b` | model name |
| `LLM_BASE_URL` | `http://127.0.0.1:11434` | Ollama or API base URL |
| `LLM_NUM_CTX` | `16384` | Ollama context window (the default is too small for tool schemas + results) |
| `GITHUB_TOKEN` | empty | empty = GitHub tools off, code questions still work |
| `GITHUB_MCP_MODE` | `stdio` | `stdio` (binary), `docker`, `remote` (GitHub-hosted), `http` (self-hosted) |
| `GITHUB_TOOLS` | 9 read tools | JSON list, e.g. `["list_issues","issue_read"]` |
| `MAX_TOOL_CALLS` | `8` | tool budget per question |
| `MAX_TOOL_OUTPUT_CHARS` | `6000` | each tool result is cut to this size |
| `CODEBASE_CACHE_TTL_S` / `CODEBASE_MAX_REPO_MB` | `3600` / `300` | clone cache lifetime and size limit |
| `API_KEY` | empty | if set, requests need the `X-API-Key` header |

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv` / `git` not found | Reopen the terminal after installing |
| `github: unavailable (FileNotFoundError ...)` | `GITHUB_MCP_COMMAND` doesn't point at the binary: check `tools\github-mcp-server.exe --version` (Step 5) |
| `github: set GITHUB_TOKEN to enable` | Add the token to `.env` (Step 4) |
| GitHub tool results say `401` or `403` | Token wrong or expired: create a new one (Step 4) |
| `codebase` tools fail with `git is not installed` | Install Git (Step 1) and reopen the terminal |
| `Repository not found, or it is private` | Only public repositories can be cloned; check the spelling `owner/name` |
| `503 LLM backend unavailable` | Ollama not running (`ollama list`) or wrong `LLM_BASE_URL` / API key |
| Answers are slow | Normal for a 3B model on a CPU: use Groq (Step 3B), or lower `MAX_TOOL_CALLS` |
| The model keeps picking the wrong tool | Small-model limit: try `qwen3:8b` or Groq; check the trace in the UI |
| Behind a company proxy: clone or GitHub calls fail | `HTTPS_PROXY`, `NO_PROXY` and CA variables are passed to both MCP servers; set them in the terminal before starting |
| Windows: MCP servers fail with `NotImplementedError` | You started uvicorn with `--reload` or `--workers`: uvicorn then uses an event loop that can't start subprocesses. Start it without those flags |
| `Remote end closed` / every request hangs on Windows | Use `http://127.0.0.1:...` rather than `localhost` (IPv6 lookup first) |
| `remote` mode returns 401/403 | GitHub's hosted MCP server may require Copilot access for your account; use `stdio` or `docker` |
