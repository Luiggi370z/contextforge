"""Tracing helpers must be safe no-ops when Langfuse is disabled."""

from __future__ import annotations

import pytest

from app.core.tracing import get_tracer, traced_span


def test_tracer_is_none_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    from app.core.config import get_settings

    get_settings.cache_clear()
    get_tracer.cache_clear()
    assert get_tracer() is None
    get_settings.cache_clear()
    get_tracer.cache_clear()


@pytest.mark.asyncio
async def test_traced_span_is_passthrough_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    from app.core.config import get_settings

    get_settings.cache_clear()
    get_tracer.cache_clear()
    async with traced_span("retrieve") as span:
        assert span is None
    get_settings.cache_clear()
    get_tracer.cache_clear()
