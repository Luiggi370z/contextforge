"""FastEmbed BM25 sparse embeddings for Qdrant hybrid search.

The model is loaded once and cached. ``embed`` runs off the event loop so async
callers (ingest, retrieval) do not block on CPU work — mirrors
``app.retrieval.embeddings.embed_texts_async``.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

import structlog

from app.core.config import get_settings

try:
    from fastembed import SparseTextEmbedding
except ImportError:  # optional ``ml`` extra
    SparseTextEmbedding = None  # type: ignore[misc, assignment]

log = structlog.get_logger(__name__)

SparseVectorTuple = tuple[list[int], list[float]]


@lru_cache(maxsize=1)
def _sparse_model():
    if SparseTextEmbedding is None:
        raise ImportError(
            "fastembed is required for sparse embeddings. Install with: uv sync --extra ml"
        )
    settings = get_settings()
    log.info("loading_sparse_model", model=settings.sparse_embedding_model)
    return SparseTextEmbedding(model_name=settings.sparse_embedding_model)


def _to_tuples(embeddings) -> list[SparseVectorTuple]:  # noqa: ANN001
    result: list[SparseVectorTuple] = []
    for emb in embeddings:
        indices = [int(i) for i in emb.indices.tolist()]
        values = [float(v) for v in emb.values.tolist()]
        result.append((indices, values))
    return result


def embed_sparse(texts: list[str]) -> list[SparseVectorTuple]:
    if not texts:
        return []
    model = _sparse_model()
    return _to_tuples(model.embed(texts))


async def embed_sparse_async(texts: list[str]) -> list[SparseVectorTuple]:
    if not texts:
        return []
    model = _sparse_model()
    embeddings = await asyncio.to_thread(lambda: list(model.embed(texts)))
    return _to_tuples(embeddings)
