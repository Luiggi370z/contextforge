import uuid
from unittest.mock import AsyncMock

import pytest

from app.api.v1.threads.service import ThreadService
from app.core.exceptions import ThreadNotFoundError


@pytest.mark.asyncio
async def test_delete_thread_raises_when_missing():
    repository = AsyncMock()
    repository.delete = AsyncMock(return_value=False)
    service = ThreadService(repository)
    session = AsyncMock()
    thread_id = uuid.uuid4()

    with pytest.raises(ThreadNotFoundError):
        await service.delete_thread(session, thread_id)

    repository.delete.assert_awaited_once_with(session, thread_id)


@pytest.mark.asyncio
async def test_delete_thread_succeeds_when_present():
    repository = AsyncMock()
    repository.delete = AsyncMock(return_value=True)
    service = ThreadService(repository)
    session = AsyncMock()
    thread_id = uuid.uuid4()

    await service.delete_thread(session, thread_id)

    repository.delete.assert_awaited_once_with(session, thread_id)
