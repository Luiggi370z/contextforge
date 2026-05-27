from __future__ import annotations

import uuid
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.constants import RRF_RANK_CONSTANT
from app.db.models import Chunk
from app.retrieval.qdrant_store import QdrantStore


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float


def reciprocal_rank_fusion(
    rank_lists: list[list[str]], k: int = RRF_RANK_CONSTANT
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _simple_rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    q_tokens = set(query.lower().split())

    def score(chunk: RetrievedChunk) -> float:
        c_tokens = set(chunk.content.lower().split())
        overlap = len(q_tokens & c_tokens) / (len(q_tokens) + 1e-9)
        return chunk.score * 0.7 + overlap * 0.3

    return sorted(candidates, key=score, reverse=True)[:top_n]


async def bm25_search(db: AsyncSession, query: str, limit: int = 20) -> list[RetrievedChunk]:
    result = await db.execute(select(Chunk))
    chunks = list(result.scalars().all())
    if not chunks:
        return []
    corpus = [c.content for c in chunks]
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(chunks, scores, strict=False), key=lambda x: x[1], reverse=True)[:limit]
    return [
        RetrievedChunk(
            chunk_id=c.id,
            document_id=c.document_id,
            content=c.content,
            score=float(s),
        )
        for c, s in ranked
    ]


async def hybrid_retrieve(
    db: AsyncSession,
    qdrant: QdrantStore,
    query: str,
) -> list[RetrievedChunk]:
    settings = get_settings()
    dense_hits = await qdrant.dense_search(query, limit=settings.retrieval_top_k)
    sparse_hits = await bm25_search(db, query, limit=settings.retrieval_top_k)

    dense_ids = [str(h.chunk_id) for h in dense_hits]
    sparse_ids = [str(h.chunk_id) for h in sparse_hits]
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])

    by_id: dict[str, RetrievedChunk] = {}
    for hit in dense_hits + sparse_hits:
        key = str(hit.chunk_id)
        existing = by_id.get(key)
        if existing is None or hit.score > existing.score:
            by_id[key] = RetrievedChunk(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                content=hit.content,
                score=hit.score,
            )

    candidates: list[RetrievedChunk] = []
    for chunk_id, rrf_score in fused:
        if chunk_id in by_id:
            chunk = by_id[chunk_id]
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=rrf_score,
                )
            )

    return _simple_rerank(query, candidates, settings.rerank_top_n)
