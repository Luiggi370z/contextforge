"""Thread listing and detail business logic."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.threads.models import Thread
from app.api.v1.threads.repository import ThreadRepository
from app.core.exceptions import ThreadNotFoundError


class ThreadService:
    """Coordinates thread persistence."""

    def __init__(self, repository: ThreadRepository | None = None) -> None:
        self._repository = repository or ThreadRepository()

    async def list_threads(self, session: AsyncSession) -> list[Thread]:
        """Return recent threads."""
        return await self._repository.list_recent(session)

    async def create_thread(self, session: AsyncSession, *, title: str | None) -> Thread:
        """Create a new conversation thread."""
        return await self._repository.create(session, title=title)

    async def get_thread_detail(
        self, session: AsyncSession, thread_id: uuid.UUID
    ) -> Thread:
        """Return a thread with messages or raise ``ThreadNotFoundError``."""
        thread = await self._repository.get_with_messages(session, thread_id)
        if thread is None:
            raise ThreadNotFoundError(str(thread_id))
        return thread

    async def delete_thread(self, session: AsyncSession, thread_id: uuid.UUID) -> None:
        """Delete a thread or raise ``ThreadNotFoundError``."""
        deleted = await self._repository.delete(session, thread_id)
        if not deleted:
            raise ThreadNotFoundError(str(thread_id))
