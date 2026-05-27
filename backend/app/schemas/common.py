from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_env: str


class ErrorResponse(BaseModel):
    detail: str
