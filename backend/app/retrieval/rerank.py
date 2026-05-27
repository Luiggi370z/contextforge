"""Rerank fused retrieval candidates (lexical or cross-encoder)."""

from __future__ import annotations

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

log = structlog.get_logger(__name__)

_cross_encoder_model = None


def _get_cross_encoder():
    """Lazy-load cross-encoder model (optional ``ml`` extra)."""
    global _cross_encoder_model
    if _cross_encoder_model is not None:
        return _cross_encoder_model
    from sentence_transformers import CrossEncoder

    _cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder_model


def lexical_rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    """Rerank by blending RRF score with token overlap to the query.

    Example:
        >>> lexical_rerank("pto policy", [RetrievedChunk(...)], top_n=3)
    """
    query_tokens = set(query.lower().split())

    def score(chunk: RetrievedChunk) -> float:
        chunk_tokens = set(chunk.content.lower().split())
        overlap = len(query_tokens & chunk_tokens) / (
            len(query_tokens) + LEXICAL_OVERLAP_EPSILON
        )
        return (
            chunk.score * RERANK_BASE_SCORE_WEIGHT
            + overlap * RERANK_LEXICAL_OVERLAP_WEIGHT
        )

    return sorted(candidates, key=score, reverse=True)[:top_n]


def cross_encoder_rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    """Rerank with a cross-encoder; falls back to lexical if the model is unavailable."""
    if not candidates:
        return []
    try:
        model = _get_cross_encoder()
    except ImportError:
        log.warning("cross_encoder_unavailable", fallback=RERANK_BACKEND_LEXICAL)
        return lexical_rerank(query, candidates, top_n)

    pairs = [(query, chunk.content) for chunk in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.content,
            score=float(score),
        )
        for chunk, score in ranked[:top_n]
    ]


def rerank_candidates(
    query: str, candidates: list[RetrievedChunk], top_n: int | None = None
) -> list[RetrievedChunk]:
    """Apply configured reranker and return top-N chunks."""
    settings = get_settings()
    limit = top_n if top_n is not None else settings.rerank_top_n
    backend = settings.rerank_backend
    if backend == RERANK_BACKEND_CROSS_ENCODER:
        return cross_encoder_rerank(query, candidates, limit)
    return lexical_rerank(query, candidates, limit)
