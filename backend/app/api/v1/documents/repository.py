"""Persistence access for documents."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.documents.models import Document, IngestionJob
from app.core.constants import DOCUMENT_LIST_LIMIT, INGESTION_JOB_QUEUED


class DocumentRepository:
    """SQLAlchemy repository for ``Document`` rows."""

    async def count(self, session: AsyncSession) -> int:
        """Return total document count."""
        total = await session.scalar(select(func.count()).select_from(Document))
        return int(total or 0)

    async def list_recent(self, session: AsyncSession) -> list[Document]:
        """List documents ordered by newest first (capped)."""
        result = await session.execute(
            select(Document).order_by(Document.created_at.desc()).limit(DOCUMENT_LIST_LIMIT)
        )
        return list(result.scalars().all())

    async def add(self, session: AsyncSession, document: Document) -> Document:
        """Persist a new document row (flush only; caller commits)."""
        session.add(document)
        await session.flush()
        return document

    async def get_by_id(self, session: AsyncSession, document_id: uuid.UUID) -> Document | None:
        """Fetch one document by primary key or ``None``."""
        return await session.get(Document, document_id)

    async def create_job(self, session: AsyncSession, *, filename: str) -> IngestionJob:
        """Create a queued ingestion job row (flush only; caller commits)."""
        job = IngestionJob(filename=filename, status=INGESTION_JOB_QUEUED, progress=0)
        session.add(job)
        await session.flush()
        return job

    async def get_job(self, session: AsyncSession, job_id: uuid.UUID) -> IngestionJob | None:
        """Fetch one ingestion job by primary key or ``None``."""
        return await session.get(IngestionJob, job_id)

    async def update_job(
        self,
        session: AsyncSession,
        job_id: uuid.UUID,
        *,
        status: str | None = None,
        progress: int | None = None,
        document_id: uuid.UUID | None = None,
        error: str | None = None,
    ) -> None:
        """Patch the given fields on an ingestion job (flush only; caller commits)."""
        job = await session.get(IngestionJob, job_id)
        if job is None:
            return
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if document_id is not None:
            job.document_id = document_id
        if error is not None:
            job.error = error
        await session.flush()
