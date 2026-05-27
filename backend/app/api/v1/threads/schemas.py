import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ThreadCreate(BaseModel):
    title: str | None = Field(default=None, max_length=256)


class ThreadResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ThreadDetailResponse(ThreadResponse):
    messages: list[MessageResponse] = Field(default_factory=list)
