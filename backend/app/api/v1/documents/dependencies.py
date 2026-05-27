"""FastAPI dependencies for document routes."""

from functools import lru_cache

from app.api.v1.documents.service import DocumentService
from app.retrieval.qdrant_store import QdrantStore, get_qdrant_store


@lru_cache
def get_document_service() -> DocumentService:
    """Singleton document service for request handlers."""
    return DocumentService()


def get_qdrant() -> QdrantStore:
    """Qdrant client used during ingestion."""
    # TODO(retrieval-backend): replace with get_vector_store() from app.retrieval.factory
    return get_qdrant_store()
