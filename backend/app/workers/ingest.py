"""ARQ task: ingest a stored upload off the request thread."""

from __future__ import annotations

import uuid

import structlog

from app.api.v1.documents.repository import DocumentRepository
from app.core.constants import (
    INGESTION_JOB_COMPLETED,
    INGESTION_JOB_FAILED,
    INGESTION_JOB_PROCESSING,
)
from app.db.session import async_session_factory
from app.ingestion.loaders import load_document
from app.ingestion.service import ingest_document_blocks
from app.retrieval.qdrant_store import get_qdrant_store

log = structlog.get_logger(__name__)


async def ingest_document_task(
    ctx: dict,
    *,
    job_id: str,
    filename: str,
    content_type: str,
    raw_bytes: bytes,
) -> str:
    """Load + ingest the uploaded bytes; update the ingestion_jobs row throughout."""
    repository = DocumentRepository()
    job_uuid = uuid.UUID(job_id)
    qdrant = get_qdrant_store()
    async with async_session_factory() as session:
        await repository.update_job(
            session, job_uuid, status=INGESTION_JOB_PROCESSING, progress=10
        )
        await session.commit()
        try:
            blocks = load_document(
                filename=filename, raw_bytes=raw_bytes, content_type=content_type
            )
            document = await ingest_document_blocks(
                session, qdrant, filename=filename, blocks=blocks, content_type=content_type
            )
            await repository.update_job(
                session,
                job_uuid,
                status=INGESTION_JOB_COMPLETED,
                progress=100,
                document_id=document.id,
            )
            await session.commit()
            log.info("ingest_job_complete", job_id=job_id, document_id=str(document.id))
            return str(document.id)
        except Exception as error:
            await repository.update_job(
                session, job_uuid, status=INGESTION_JOB_FAILED, error=str(error)
            )
            await session.commit()
            log.exception("ingest_job_failed", job_id=job_id, error=str(error))
            raise
