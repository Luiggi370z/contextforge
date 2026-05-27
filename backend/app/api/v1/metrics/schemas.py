from pydantic import BaseModel


class MetricsResponse(BaseModel):
    queries_total: int
    ingests_total: int
