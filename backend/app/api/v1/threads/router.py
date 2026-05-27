"""HTTP routes for conversation threads."""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.threads.dependencies import get_thread_service
from app.api.v1.threads.schemas import ThreadCreate, ThreadDetailResponse, ThreadResponse
from app.api.v1.threads.service import ThreadService
from app.core.exceptions import ThreadNotFoundError, domain_error_to_app_exception
from app.db.session import get_db

router = APIRouter()


@router.get("", response_model=list[ThreadResponse])
async def list_threads(
    session: AsyncSession = Depends(get_db),
    thread_service: ThreadService = Depends(get_thread_service),
) -> list[ThreadResponse]:
    """List recent threads."""
    threads = await thread_service.list_threads(session)
    return [ThreadResponse.model_validate(thread) for thread in threads]


@router.post("", response_model=ThreadResponse, status_code=201)
async def create_thread(
    body: ThreadCreate,
    session: AsyncSession = Depends(get_db),
    thread_service: ThreadService = Depends(get_thread_service),
) -> ThreadResponse:
    """Create a new thread."""
    thread = await thread_service.create_thread(session, title=body.title)
    return ThreadResponse.model_validate(thread)


@router.get("/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread(
    thread_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    thread_service: ThreadService = Depends(get_thread_service),
) -> ThreadDetailResponse:
    """Return a thread with its messages."""
    try:
        thread = await thread_service.get_thread_detail(session, thread_id)
    except ThreadNotFoundError as error:
        raise domain_error_to_app_exception(error) from error
    return ThreadDetailResponse.model_validate(thread)
