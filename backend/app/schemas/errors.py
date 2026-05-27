from pydantic import Field

from app.schemas.base import BaseResponse


class ErrorResponse(BaseResponse):
    """Standard JSON error body for API clients."""

    detail: str
    correlation_id: str | None = Field(default=None)
