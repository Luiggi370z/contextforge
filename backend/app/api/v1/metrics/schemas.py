from app.schemas.base import BaseResponse


class MetricsResponse(BaseResponse):
    queries_total: int
    ingests_total: int
