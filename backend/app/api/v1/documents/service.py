"""Document ingestion and listing business logic."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.documents.models import DocumentListResult
from app.api.v1.documents.repository import DocumentRepository
from app.core.constants import (
    DEFAULT_TEXT_CONTENT_TYPE,
    DEFAULT_UPLOAD_FILENAME,
    INGESTION_JOB_COMPLETED,
    INGESTION_JOB_FAILED,
)
from app.core.exceptions import IngestionError
from app.db.models import Document, IngestionJob
from app.ingestion.loaders import load_document
from app.ingestion.service import ingest_document_blocks, ingest_document_text
from app.retrieval.qdrant_store import QdrantStore, get_qdrant_store

log = structlog.get_logger(__name__)


class DocumentService:
    """Coordinates repository, Qdrant, and ingestion pipeline."""

    def __init__(
        self,
        repository: DocumentRepository | None = None,
    ) -> None:
        self._repository = repository or DocumentRepository()

    async def list_documents(self, session: AsyncSession) -> DocumentListResult:
        """List recent documents with total count."""
        items = await self._repository.list_recent(session)
        total = await self._repository.count(session)
        return DocumentListResult(items=items, total=total)

    async def ingest_text(
        self,
        session: AsyncSession,
        qdrant: QdrantStore,
        *,
        filename: str,
        content: str,
        content_type: str = DEFAULT_TEXT_CONTENT_TYPE,
    ) -> Document:
        """Ingest plain text as a document (chunk, embed, upsert)."""
        try:
            document = await ingest_document_text(
                session,
                qdrant,
                filename=filename,
                content=content,
                content_type=content_type,
            )
            log.info("document_ingested", document_id=str(document.id), filename=filename)
            return document
        except Exception as error:
            log.exception("ingest_failed", filename=filename, error=str(error))
            raise IngestionError(str(error)) from error

    async def ingest_upload(
        self,
        session: AsyncSession,
        qdrant: QdrantStore,
        *,
        filename: str | None,
        raw_bytes: bytes,
        content_type: str | None,
    ) -> Document:
        """Load uploaded bytes by type (PDF/MD/TXT) and ingest."""
        resolved_name = filename or DEFAULT_UPLOAD_FILENAME
        resolved_type = content_type or DEFAULT_TEXT_CONTENT_TYPE
        try:
            blocks = await asyncio.to_thread(
                load_document,
                filename=resolved_name,
                raw_bytes=raw_bytes,
                content_type=resolved_type,
            )
            document = await ingest_document_blocks(
                session,
                qdrant,
                filename=resolved_name,
                blocks=blocks,
                content_type=resolved_type,
            )
            log.info("document_ingested", document_id=str(document.id), filename=resolved_name)
            return document
        except Exception as error:
            log.exception("ingest_failed", filename=resolved_name, error=str(error))
            raise IngestionError(str(error)) from error

    async def enqueue_upload(
        self,
        session: AsyncSession,
        arq_pool: Any,
        *,
        filename: str | None,
        raw_bytes: bytes,
        content_type: str | None,
    ) -> IngestionJob:
        """Create a queued ingestion_jobs row and enqueue the ARQ worker task.

        When no pool is available (Redis down), fall back to a real synchronous
        ingest so the demo still works: load + ingest the bytes, then mark the
        job complete with the new document id.
        """
        resolved_name = filename or DEFAULT_UPLOAD_FILENAME
        resolved_type = content_type or DEFAULT_TEXT_CONTENT_TYPE
        job = await self._repository.create_job(session, filename=resolved_name)
        await session.commit()

        if arq_pool is not None:
            await arq_pool.enqueue_job(
                "ingest_document_task",
                job_id=str(job.id),
                filename=resolved_name,
                content_type=resolved_type,
                raw_bytes=raw_bytes,
            )
            return job

        # Synchronous fallback — keeps the demo working without Redis.
        try:
            blocks = await asyncio.to_thread(
                load_document,
                filename=resolved_name,
                raw_bytes=raw_bytes,
                content_type=resolved_type,
            )
            document = await ingest_document_blocks(
                session,
                get_qdrant_store(),
                filename=resolved_name,
                blocks=blocks,
                content_type=resolved_type,
            )
            await self._repository.update_job(
                session,
                job.id,
                status=INGESTION_JOB_COMPLETED,
                progress=100,
                document_id=document.id,
            )
            await session.commit()
            log.info(
                "document_ingested_sync",
                document_id=str(document.id),
                filename=resolved_name,
            )
        except Exception as error:
            await self._repository.update_job(
                session, job.id, status=INGESTION_JOB_FAILED, error=str(error)
            )
            await session.commit()
            log.exception("ingest_sync_failed", filename=resolved_name, error=str(error))
            raise IngestionError(str(error)) from error
        return job

    async def get_job(self, session: AsyncSession, job_id: uuid.UUID) -> IngestionJob | None:
        """Fetch one ingestion job by id (or ``None``)."""
        return await self._repository.get_job(session, job_id)
