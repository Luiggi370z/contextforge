"""Root API router wiring all versioned sub-routers."""

from fastapi import APIRouter

from app.api.v1 import documents, health, metrics, query, threads
from app.core.constants import API_V1_PREFIX

def build_api_router() -> APIRouter:
    """Register all HTTP routers under the v1 prefix.

    Example:
        ``app.include_router(build_api_router())`` mounts ``/v1/health``, etc.
    """
    version_one = APIRouter(prefix=API_V1_PREFIX)
    version_one.include_router(health.router, tags=["health"])
    version_one.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
    version_one.include_router(documents.router, prefix="/documents", tags=["documents"])
    version_one.include_router(threads.router, prefix="/threads", tags=["threads"])
    version_one.include_router(query.router, prefix="/query", tags=["query"])
    return version_one
