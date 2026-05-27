from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard JSON error body for API clients."""

    detail: str
    correlation_id: str | None = Field(default=None)
