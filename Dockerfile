# syntax=docker/dockerfile:1
# Two images from one file:
#   docker build --target api -t repo-scout-api .
#   docker build --target ui  -t repo-scout-ui  .

ARG GITHUB_MCP_VERSION=v2.2.0

# The official GitHub MCP server image: we only take its (static) binary.
FROM ghcr.io/github/github-mcp-server:${GITHUB_MCP_VERSION} AS github-mcp

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN useradd --create-home --uid 10001 app

# --- API: FastAPI + LangGraph agent + both MCP servers (stdio subprocesses) ----
FROM base AS api
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=github-mcp /server/github-mcp-server /usr/local/bin/github-mcp-server
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups
RUN mkdir -p /app/.cache/repos && chown -R app /app/.cache
ENV REPOSCOUT_IN_DOCKER=1 \
    GITHUB_MCP_MODE=stdio \
    GITHUB_MCP_COMMAND=/usr/local/bin/github-mcp-server \
    CODEBASE_CACHE_DIR=/app/.cache/repos
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "reposcout.api:app", "--host", "0.0.0.0", "--port", "8000"]

# --- UI: Streamlit only (no agent code, no LLM libraries) ------------------------
FROM base AS ui
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --only-group ui --no-install-project
COPY ui ./ui
USER app
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
