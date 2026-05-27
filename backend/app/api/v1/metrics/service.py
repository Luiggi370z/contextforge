"""In-process request counters for demo metrics."""

from app.api.v1.metrics.schemas import MetricsResponse


class MetricsService:
    """Tracks query and ingest totals (process-local)."""

    def __init__(self) -> None:
        self._queries_total = 0
        self._ingests_total = 0

    def increment_queries(self) -> None:
        self._queries_total += 1

    def increment_ingests(self) -> None:
        self._ingests_total += 1

    def snapshot(self) -> MetricsResponse:
        return MetricsResponse(
            queries_total=self._queries_total,
            ingests_total=self._ingests_total,
        )


_metrics_service = MetricsService()


def get_metrics_service() -> MetricsService:
    return _metrics_service


def increment_queries() -> None:
    get_metrics_service().increment_queries()


def increment_ingests() -> None:
    get_metrics_service().increment_ingests()
