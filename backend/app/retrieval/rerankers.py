"""Concrete Reranker implementations (delegate to app.retrieval.rerank)."""

from __future__ import annotations

from app.retrieval.models import RetrievedChunk
from app.retrieval.rerank import cross_encoder_rerank_async, lexical_rerank


class LexicalReranker:
    """Token-overlap blend reranker (synchronous, wrapped async)."""

    name = "lexical"

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return lexical_rerank(query, candidates, top_n)


class CrossEncoderReranker:
    """Cross-encoder reranker with timeout + graceful fallback (Wave 0 PR1)."""

    name = "cross_encoder"

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return await cross_encoder_rerank_async(query, candidates, top_n)
