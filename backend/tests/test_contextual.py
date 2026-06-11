"""Contextualization: heuristic returns the static prefix (deterministic CI)."""

from __future__ import annotations

import pytest

from app.llm.providers import get_llm_provider, reset_llm_provider_cache


@pytest.mark.asyncio
async def test_heuristic_contextualize_returns_static_prefix(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "heuristic")
    reset_llm_provider_cache()
    provider = get_llm_provider()

    blurb = await provider.contextualize(
        document_title="pto.md",
        section="PTO",
        chunk="Full-time staff accrue 20 PTO days per calendar year.",
        full_document="...whole doc...",
    )
    reset_llm_provider_cache()
    # Heuristic = the existing structural prefix, verbatim. No LLM, deterministic.
    assert blurb == "Document: pto.md > Section: PTO"
