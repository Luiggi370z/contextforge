"""Postgres checkpointer for LangGraph (optional at runtime)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog

from app.core.config import get_settings

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:
    AsyncPostgresSaver = None  # type: ignore[misc, assignment]

log = structlog.get_logger(__name__)


def _psycopg_url(database_url: str) -> str:
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


@asynccontextmanager
async def postgres_checkpointer() -> AsyncIterator[object | None]:
    settings = get_settings()
    if AsyncPostgresSaver is None:
        log.warning("checkpointer_unavailable", error="langgraph-checkpoint-postgres not installed")
        yield None
        return
    try:
        uri = _psycopg_url(settings.database_url)
        async with AsyncPostgresSaver.from_conn_string(uri) as saver:
            await saver.setup()
            log.info("langgraph_checkpointer_ready")
            yield saver
    except Exception as exc:
        log.warning("checkpointer_unavailable", error=str(exc))
        yield None
