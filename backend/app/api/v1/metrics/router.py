"""Prometheus-style counters for local demos."""

from fastapi import APIRouter, Depends

from app.api.v1.metrics.schemas import MetricsResponse
from app.api.v1.metrics.service import MetricsService, get_metrics_service

router = APIRouter()


@router.get("", response_model=MetricsResponse)
async def metrics(
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> MetricsResponse:
    """Return total query and ingest counts since process start."""
    return metrics_service.snapshot()
