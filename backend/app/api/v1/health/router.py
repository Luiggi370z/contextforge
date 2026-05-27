"""Health check route."""

from fastapi import APIRouter

from app.api.v1.health.schemas import HealthResponse
from app.core.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe for load balancers and compose healthchecks."""
    settings = get_settings()
    return HealthResponse(status="ok", app_env=settings.app_env)
