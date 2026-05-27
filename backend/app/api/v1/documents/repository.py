"""Persistence access for documents."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import DOCUMENT_LIST_LIMIT
from app.db.models import Document


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
