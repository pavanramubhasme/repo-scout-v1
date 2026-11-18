"""Chat model factory: local Ollama, or any OpenAI-compatible endpoint (Groq, OpenRouter, vLLM...)."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel

from .config import Settings

log = logging.getLogger(__name__)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama"}


def is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in _LOCAL_HOSTS


def resolve_base_url(url: str) -> str:
    """Inside the Docker image, 127.0.0.1 is the container itself: point at the host instead,
    so the same .env works locally and in Docker."""
    host = urlparse(url).hostname or ""
    if os.environ.get("REPOSCOUT_IN_DOCKER") == "1" and host in {"localhost", "127.0.0.1"}:
        new = url.replace(host, "host.docker.internal", 1)
        log.info("llm_base_url_rewritten from=%s to=%s", url, new)
        return new
    return url


def build_chat_model(settings: Settings) -> BaseChatModel:
    base_url = resolve_base_url(settings.llm_base_url)
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        # A company proxy (HTTP_PROXY) must never intercept calls to a local Ollama.
        client_kwargs = (
            {"trust_env": False, "timeout": settings.llm_timeout_s}
            if is_local(base_url)
            else {"timeout": settings.llm_timeout_s}
        )
        return ChatOllama(
            model=settings.llm_model,
            base_url=base_url,
            temperature=settings.llm_temperature,
            seed=settings.llm_seed,
            num_ctx=settings.llm_num_ctx,
            client_kwargs=client_kwargs,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        base_url=base_url or None,
        api_key=settings.llm_api_key or "not-needed",
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )
