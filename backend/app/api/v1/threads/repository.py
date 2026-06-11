"""Persistence access for conversation threads."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.v1.threads.models import Thread
from app.core.constants import THREAD_LIST_LIMIT


class ThreadRepository:
    """SQLAlchemy repository for ``Thread`` rows."""

    async def list_recent(self, session: AsyncSession) -> list[Thread]:
        """List threads ordered by newest first (capped)."""
        result = await session.execute(
            select(Thread).order_by(Thread.created_at.desc()).limit(THREAD_LIST_LIMIT)
        )
        return list(result.scalars().all())

    async def create(self, session: AsyncSession, *, title: str | None) -> Thread:
        """Create and persist a new thread."""
        thread = Thread(title=title)
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        return thread

    async def get_with_messages(
        self, session: AsyncSession, thread_id: uuid.UUID
    ) -> Thread | None:
        """Load a thread with messages eager-loaded."""
        result = await session.execute(
            select(Thread).where(Thread.id == thread_id).options(selectinload(Thread.messages))
        )
        return result.scalar_one_or_none()

    async def delete(self, session: AsyncSession, thread_id: uuid.UUID) -> bool:
        """Delete a thread and its messages. Returns False if the thread does not exist."""
        thread = await session.get(Thread, thread_id)
        if thread is None:
            return False
        await session.delete(thread)
        await session.commit()
        return True
