"""ARQ worker settings. Run with: arq app.workers.settings.WorkerSettings"""

from __future__ import annotations

from arq.connections import RedisSettings

from app.core.config import get_settings
from app.workers.ingest import ingest_document_task


def _redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    functions = [ingest_document_task]
    redis_settings = _redis_settings()
