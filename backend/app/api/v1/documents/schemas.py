import uuid
from dataclasses import dataclass
from datetime import datetime

from pydantic import ConfigDict, Field

from app.api.v1.documents.models import Document
from app.core.constants import DEFAULT_MARKDOWN_CONTENT_TYPE
from app.schemas.base import BaseRequest, BaseResponse


@dataclass(frozen=True)
class DocumentListResult:
    """Documents returned from a list operation (internal aggregate, not a wire type)."""

    items: list[Document]
    total: int


class DocumentResponse(BaseResponse):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    status: str
    size_bytes: int | None = None
    created_at: datetime


class DocumentListResponse(BaseResponse):
    items: list[DocumentResponse]
    total: int


class IngestTextRequest(BaseRequest):
    filename: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1)
    content_type: str = DEFAULT_MARKDOWN_CONTENT_TYPE


class IngestionJobResponse(BaseResponse):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: str
    progress: int
    document_id: uuid.UUID | None = None
    error: str | None = None
