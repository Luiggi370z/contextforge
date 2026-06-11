import uuid
from datetime import datetime

from pydantic import ConfigDict

from app.schemas.base import BaseRequest, BaseResponse


class EvalRunRequest(BaseRequest):
    """Which retrieval configs to evaluate (defaults to the standard sweep)."""

    configs: list[str] | None = None


class EvalRunResponse(BaseResponse):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    label: str
    config: dict
    metrics: dict
    row_count: int
    created_at: datetime


class EvalRunListResponse(BaseResponse):
    items: list[EvalRunResponse]
    total: int
