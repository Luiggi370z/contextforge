"""FastAPI dependencies for document routes."""

from functools import lru_cache

from fastapi import Request

from app.api.v1.documents.service import DocumentService
from app.retrieval.qdrant_store import QdrantStore, get_qdrant_store


@lru_cache
def get_document_service() -> DocumentService:
    """Singleton document service for request handlers."""
    # TODO(session-in-service-di): return DocumentService(session=Depends(get_db)) per request;
    # drop @lru_cache and pass session on each route method.
    return DocumentService()


def get_qdrant() -> QdrantStore:
    """Qdrant client used during ingestion."""
    # TODO(retrieval-backend): replace with get_vector_store() from app.retrieval.factory
    return get_qdrant_store()


async def get_arq_pool(request: Request):
    """Return the shared ARQ pool from app.state (created at startup).

    May be ``None`` when Redis is unavailable; the upload path then falls back
    to a synchronous ingest.
    """
    return getattr(request.app.state, "arq_pool", None)
