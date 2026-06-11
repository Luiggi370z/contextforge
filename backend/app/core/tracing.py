"""Optional Langfuse tracing — a no-op when disabled.

Spans wrap at the graph-runner boundary so node code stays clean. When
``langfuse_enabled`` is false (or the SDK / keys are missing) every helper here
is a zero-overhead no-op.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

import structlog

from app.core.config import get_settings

try:
    from langfuse import Langfuse
except ImportError:  # optional ``obs`` extra
    Langfuse = None  # type: ignore[misc, assignment]

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_tracer():
    """Return a configured Langfuse client, or None when disabled/unavailable."""
    settings = get_settings()
    if not settings.langfuse_enabled or Langfuse is None:
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        log.warning("langfuse_enabled_but_keys_missing")
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


@asynccontextmanager
async def traced_span(name: str, **metadata):
    """Async context manager yielding a Langfuse span, or None when disabled.

    Example:
        async with traced_span("rag_query", thread_id=tid):
            ...
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_observation(name=name) as span:
        if metadata:
            span.update(metadata=metadata)
        yield span


def flush_tracer() -> None:
    tracer = get_tracer()
    if tracer is not None:
        tracer.flush()
