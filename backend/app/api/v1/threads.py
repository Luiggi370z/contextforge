import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Thread
from app.db.session import get_db
from app.schemas.threads import ThreadCreate, ThreadDetailResponse, ThreadResponse

router = APIRouter()


@router.get("", response_model=list[ThreadResponse])
async def list_threads(db: AsyncSession = Depends(get_db)) -> list[ThreadResponse]:
    result = await db.execute(select(Thread).order_by(Thread.created_at.desc()).limit(50))
    return [ThreadResponse.model_validate(t) for t in result.scalars().all()]


@router.post("", response_model=ThreadResponse, status_code=201)
async def create_thread(body: ThreadCreate, db: AsyncSession = Depends(get_db)) -> ThreadResponse:
    thread = Thread(title=body.title)
    db.add(thread)
    await db.commit()
    await db.refresh(thread)
    return ThreadResponse.model_validate(thread)


@router.get("/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread(
    thread_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> ThreadDetailResponse:
    result = await db.execute(
        select(Thread).where(Thread.id == thread_id).options(selectinload(Thread.messages))
    )
    thread = result.scalar_one_or_none()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadDetailResponse.model_validate(thread)
