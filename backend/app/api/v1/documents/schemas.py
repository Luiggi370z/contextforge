import uuid
from datetime import datetime

from pydantic import ConfigDict, Field

from app.core.constants import DEFAULT_MARKDOWN_CONTENT_TYPE
from app.schemas.base import BaseRequest, BaseResponse


class DocumentResponse(BaseResponse):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    status: str
    created_at: datetime


class DocumentListResponse(BaseResponse):
    items: list[DocumentResponse]
    total: int


class IngestTextRequest(BaseRequest):
    filename: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1)
    content_type: str = DEFAULT_MARKDOWN_CONTENT_TYPE
