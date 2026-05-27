import uuid
from datetime import datetime

from pydantic import ConfigDict, Field

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


class ThreadDetailResponse(ThreadResponse):
    messages: list[MessageResponse] = Field(default_factory=list)
