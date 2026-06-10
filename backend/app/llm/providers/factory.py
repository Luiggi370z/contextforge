"""Pick the active ``LLMProvider`` from ``settings.llm_provider``."""

from __future__ import annotations

import structlog

from app.core.config import get_settings
from app.llm.providers.base import LLMProvider
from app.llm.providers.heuristic import HeuristicProvider

log = structlog.get_logger(__name__)


_cached_provider: LLMProvider | None = None
_cached_provider_key: tuple | None = None


def _provider_key() -> tuple:
    settings = get_settings()
    return (
        settings.llm_provider,
        settings.ollama_base_url,
        settings.ollama_model,
        settings.openai_api_key or "",
        settings.openai_model,
    )


def reset_llm_provider_cache() -> None:
    """Drop the cached provider; tests use this when they change settings."""
    global _cached_provider, _cached_provider_key
    _cached_provider = None
    _cached_provider_key = None


def get_llm_provider() -> LLMProvider:
    """Build (or return cached) provider matching the current settings."""
    global _cached_provider, _cached_provider_key

    settings = get_settings()
    key = _provider_key()
    if _cached_provider is not None and _cached_provider_key == key:
        return _cached_provider

    provider_name = settings.llm_provider

    if provider_name == "ollama":
        from app.llm.providers.ollama import OllamaProvider

        provider: LLMProvider = OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    elif provider_name == "openai":
        if not settings.openai_api_key:
            log.warning(
                "llm_provider_missing_openai_key_falling_back_to_heuristic",
            )
            provider = HeuristicProvider()
        else:
            from app.llm.providers.openai import OpenAIProvider

            provider = OpenAIProvider(model=settings.openai_model)
    else:
        provider = HeuristicProvider()

    _cached_provider = provider
    _cached_provider_key = key
    log.info("llm_provider_selected", name=provider.name)
    return provider
