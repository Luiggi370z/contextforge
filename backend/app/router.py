"""Root API router wiring all versioned sub-routers."""

from fastapi import APIRouter

# Import each router from its submodule directly. Domain ``__init__`` files are kept
# side-effect-free so importing ``<domain>.model`` (ORM) from core code never drags in
# the HTTP/router stack — which would otherwise create an import cycle.
from app.api.v1.documents.router import router as documents_router
from app.api.v1.eval.router import router as eval_router
from app.api.v1.health.router import router as health_router
from app.api.v1.metrics.router import router as metrics_router
from app.api.v1.query.router import router as query_router
from app.api.v1.threads.router import router as threads_router
from app.core.constants import API_V1_PREFIX


def build_api_router() -> APIRouter:
    """Register all HTTP routers under the v1 prefix.

    Example:
        ``app.include_router(build_api_router())`` mounts ``/v1/health``, etc.
    """
    version_one = APIRouter(prefix=API_V1_PREFIX)
    version_one.include_router(health_router, tags=["health"])
    version_one.include_router(metrics_router, prefix="/metrics", tags=["metrics"])
    version_one.include_router(documents_router, prefix="/documents", tags=["documents"])
    version_one.include_router(threads_router, prefix="/threads", tags=["threads"])
    version_one.include_router(query_router, prefix="/query", tags=["query"])
    version_one.include_router(eval_router, prefix="/eval", tags=["eval"])
    return version_one
