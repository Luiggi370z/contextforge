from app.schemas.base import BaseResponse


class HealthResponse(BaseResponse):
    status: str
    app_env: str
