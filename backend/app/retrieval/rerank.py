"""Rerank fused retrieval candidates (lexical or cross-encoder).

The reranker is the last stage to write ``score``; downstream code consults
``RetrievedChunk.ranking_score()`` and gets the rerank score back. Stage-local
scores (``dense_score``, ``sparse_score``, ``rrf_score``) are preserved so the
debug panel and audits can show provenance.
"""

from __future__ import annotations

import asyncio
from typing import cast

import structlog

from app.core.config import get_settings
from app.core.constants import (
    CROSS_ENCODER_MODEL_NAME,
    LEXICAL_OVERLAP_EPSILON,
    RERANK_BACKEND_CROSS_ENCODER,
    RERANK_BACKEND_LEXICAL,
    RERANK_BASE_SCORE_WEIGHT,
    RERANK_LEXICAL_OVERLAP_WEIGHT,
)
from app.retrieval.models import RetrievedChunk

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # optional ``ml`` extra
    CrossEncoder = None  # type: ignore[misc, assignment]

log = structlog.get_logger(__name__)

_cross_encoder_model = None


def _get_cross_encoder():
    """Lazy-load cross-encoder model (optional ``ml`` extra)."""
    global _cross_encoder_model
    if _cross_encoder_model is not None:
        return _cross_encoder_model
    if CrossEncoder is None:
        raise ImportError(
            "sentence-transformers is required for cross-encoder rerank. "
            "Install with: uv sync --extra ml"
        )

    _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder_model


def _replace_score(chunk: RetrievedChunk, rerank_score: float) -> RetrievedChunk:
    """Return a chunk with ``rerank_score`` filled in and ``score`` set to it."""
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        content=chunk.content,
        score=rerank_score,
        metadata=chunk.metadata or {},
        dense_score=chunk.dense_score,
        sparse_score=chunk.sparse_score,
        rrf_score=chunk.rrf_score,
        rerank_score=rerank_score,
    )


def lexical_rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    """Rerank by blending the prior stage score with query/chunk token overlap."""
    query_tokens = set(query.lower().split())

    def blend(chunk: RetrievedChunk) -> float:
        chunk_tokens = set(chunk.content.lower().split())
        overlap = len(query_tokens & chunk_tokens) / (
            len(query_tokens) + LEXICAL_OVERLAP_EPSILON
        )
        base = chunk.ranking_score()
        return base * RERANK_BASE_SCORE_WEIGHT + overlap * RERANK_LEXICAL_OVERLAP_WEIGHT

    ranked = sorted(candidates, key=blend, reverse=True)[:top_n]
    return [_replace_score(chunk, blend(chunk)) for chunk in ranked]


async def cross_encoder_rerank_async(
    query: str, candidates: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    """Run the cross-encoder off the event loop and merge scores back in.

    Falls back to :func:`lexical_rerank` when the optional ``ml`` extra is
    not installed.
    """
    if not candidates:
        return []
    try:
        model = _get_cross_encoder()
    except ImportError:
        log.warning("cross_encoder_unavailable", fallback=RERANK_BACKEND_LEXICAL)
        return lexical_rerank(query, candidates, top_n)

    pairs = [(query, chunk.content) for chunk in candidates]
    scores = await asyncio.to_thread(model.predict, pairs)
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    return [_replace_score(chunk, float(score)) for chunk, score in ranked[:top_n]]


def cross_encoder_rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    """Sync wrapper kept for code paths that have not been awaited yet (e.g. eval)."""
    return cast(
        list[RetrievedChunk],
        asyncio.get_event_loop().run_until_complete(
            cross_encoder_rerank_async(query, candidates, top_n)
        ),
    )


def rerank_candidates(
    query: str, candidates: list[RetrievedChunk], top_n: int | None = None
) -> list[RetrievedChunk]:
    """Apply the configured reranker and return the top-``top_n`` chunks.

    The lexical backend is synchronous and safe to call inline. The cross-encoder
    is invoked through :func:`rerank_candidates_async` so the graph nodes get the
    non-blocking version automatically.
    """
    settings = get_settings()
    limit = top_n if top_n is not None else settings.rerank_top_n
    backend = settings.rerank_backend
    if backend == RERANK_BACKEND_CROSS_ENCODER:
        return cross_encoder_rerank(query, candidates, limit)
    return lexical_rerank(query, candidates, limit)


async def rerank_candidates_async(
    query: str, candidates: list[RetrievedChunk], top_n: int | None = None
) -> list[RetrievedChunk]:
    """Async-safe rerank: cross-encoder runs in a worker thread."""
    settings = get_settings()
    limit = top_n if top_n is not None else settings.rerank_top_n
    if settings.rerank_backend == RERANK_BACKEND_CROSS_ENCODER:
        return await cross_encoder_rerank_async(query, candidates, limit)
    return lexical_rerank(query, candidates, limit)
