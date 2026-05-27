"""Pluggable retrieval backends (Qdrant today; Postgres pgvector+FTS planned).

TODO(retrieval-backend): Implement and wire:
  - ``PostgresVectorStore`` — dense ANN via pgvector on ``chunks.embedding``
  - ``postgres_fts_search`` — sparse leg via ``chunks.content_tsv`` + GIN index
  - ``get_vector_store()`` / ``hybrid_retrieve_configured()`` — branch on
    ``settings.retrieval_backend`` (``RETRIEVAL_BACKEND_QDRANT`` | ``RETRIEVAL_BACKEND_POSTGRES``)
  - Alembic migration: enable pgvector, add embedding + tsvector columns
  - Docker: optional ``postgres:16`` image with pgvector; Qdrant service skippable when postgres mode

See ARCHITECTURE.md § Pluggable retrieval backends.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.constants import RETRIEVAL_BACKEND_POSTGRES, RETRIEVAL_BACKEND_QDRANT
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.models import RetrievedChunk
from app.retrieval.qdrant_store import QdrantStore, get_qdrant_store


def get_vector_store() -> QdrantStore:
    """Return the active dense-vector store for the configured retrieval backend."""
    settings = get_settings()
    if settings.retrieval_backend == RETRIEVAL_BACKEND_POSTGRES:
        # TODO(retrieval-backend): return PostgresVectorStore(session) or pool-bound facade
        raise NotImplementedError(
            f"{RETRIEVAL_BACKEND_POSTGRES} retrieval backend is not implemented yet; "
            f"use {RETRIEVAL_BACKEND_QDRANT}"
        )
    return get_qdrant_store()


async def hybrid_retrieve_configured(
    session: AsyncSession,
    query: str,
    qdrant: QdrantStore | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid retrieval using the backend selected in settings."""
    settings = get_settings()
    if settings.retrieval_backend == RETRIEVAL_BACKEND_POSTGRES:
        # TODO(retrieval-backend): dense via pgvector + sparse via Postgres FTS, then RRF + rerank
        raise NotImplementedError(
            f"{RETRIEVAL_BACKEND_POSTGRES} retrieval backend is not implemented yet"
        )
    store = qdrant if qdrant is not None else get_qdrant_store()
    return await hybrid_retrieve(session, store, query)


# TODO(retrieval-backend-ui): Optional Postgres retrieval path selectable from the React UI.
#
# End-to-end work (after postgres vector store + FTS are implemented above):
#   1. API — GET/PATCH /v1/settings (or /v1/config) exposing ``retrieval_backend`` with
#      validation against RETRIEVAL_BACKEND_QDRANT | RETRIEVAL_BACKEND_POSTGRES; persist per
#      deployment (env override) or per-user/session if product needs it.
#   2. Request scope — pass chosen backend on POST /v1/query (header or body field) so a
#      single running API can serve both modes without restart; factory reads override then
#      falls back to Settings.retrieval_backend.
#   3. UI — settings control in the chat app (e.g. DebugPanel or Settings drawer):
#      toggle "Qdrant hybrid" vs "Postgres (pgvector + FTS)"; store preference in localStorage;
#      send with each query; show active backend in debug metadata.
#   4. Ingest — document upload must use the same backend as search (re-ingest or dual-write
#      policy documented in ARCHITECTURE.md).
#   5. Compose — when UI selects postgres, health check should verify pgvector extension;
#      hide/disable Qdrant-only options if qdrant container is not running.
