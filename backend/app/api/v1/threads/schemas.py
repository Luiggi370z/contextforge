from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import ConfigDict, Field

from app.api.v1.query.schemas import Citation, QueryMetadata
from app.db.models import Message, Thread
from app.schemas.base import BaseRequest, BaseResponse


class ThreadCreate(BaseRequest):
    title: str | None = Field(default=None, max_length=256)


class ThreadResponse(BaseResponse):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: datetime


class MessageResponse(BaseResponse):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    metadata: QueryMetadata | None = None
    citations: list[Citation] = Field(default_factory=list)


class ThreadDetailResponse(ThreadResponse):
    messages: list[MessageResponse] = Field(default_factory=list)

    @classmethod
    def from_thread(cls, thread: Thread) -> ThreadDetailResponse:
        """Build API response including citations stored on assistant message metadata."""
        ordered_messages = sorted(thread.messages, key=lambda row: row.created_at)
        return cls(
            id=thread.id,
            title=thread.title,
            created_at=thread.created_at,
            messages=[message_response_from_orm(message) for message in ordered_messages],
        )


def message_response_from_orm(message: Message) -> MessageResponse:
    """Split persisted metadata JSON into query metadata and citations list."""
    stored = dict(message.metadata_ or {})
    citations_payload = stored.pop("citations", [])
    metadata = QueryMetadata.model_validate(stored) if stored else None
    citations = [
        Citation.model_validate(item)
        for item in citations_payload
        if isinstance(item, dict)
    ]
    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        metadata=metadata,
        citations=citations,
    )
