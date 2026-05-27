from __future__ import annotations

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.constants import RRF_RANK_CONSTANT
from app.db.models import Chunk
from app.retrieval.models import RetrievedChunk
from app.retrieval.qdrant_store import QdrantStore, VectorRecord
from app.retrieval.rerank import rerank_candidates


def reciprocal_rank_fusion(
    rank_lists: list[list[str]], k: int = RRF_RANK_CONSTANT
) -> list[tuple[str, float]]:
    """Merge ranked id lists with Reciprocal Rank Fusion.

    Example:
        >>> reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0][0]
        'b'
    """
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def merge_retrieval_hits(
    dense_hits: list[VectorRecord],
    sparse_hits: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Fuse dense and sparse hits with RRF scores."""
    dense_ids = [str(hit.chunk_id) for hit in dense_hits]
    sparse_ids = [str(hit.chunk_id) for hit in sparse_hits]
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])

    by_id: dict[str, RetrievedChunk] = {}
    for hit in dense_hits + sparse_hits:
        key = str(hit.chunk_id)
        content = hit.content
        document_id = hit.document_id
        score = hit.score
        existing = by_id.get(key)
        if existing is None or score > existing.score:
            by_id[key] = RetrievedChunk(
                chunk_id=hit.chunk_id,
                document_id=document_id,
                content=content,
                score=score,
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
    return candidates


# TODO(retrieval-backend): When RETRIEVAL_BACKEND=postgres, replace with Postgres FTS
# (tsvector + ts_rank / websearch_to_tsquery) instead of loading all chunks for BM25Okapi.


async def bm25_search(
    session: AsyncSession, query: str, limit: int | None = None
) -> list[RetrievedChunk]:
    """Rank chunks in Postgres with BM25 over tokenized content."""
    settings = get_settings()
    search_limit = limit if limit is not None else settings.retrieval_top_k
    result = await session.execute(select(Chunk))
    chunks = list(result.scalars().all())
    if not chunks:
        return []
    corpus = [chunk.content for chunk in chunks]
    tokenized = [document.lower().split() for document in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(chunks, scores, strict=False), key=lambda pair: pair[1], reverse=True)[
        :search_limit
    ]
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            score=float(score),
        )
        for chunk, score in ranked
    ]


async def hybrid_retrieve(
    session: AsyncSession,
    qdrant: QdrantStore,
    query: str,
) -> list[RetrievedChunk]:
    """Dense (Qdrant) + sparse (BM25) retrieval, RRF fusion, then rerank.

    TODO(retrieval-backend): Prefer ``hybrid_retrieve_configured()`` from factory.py
    once Postgres mode exists; both legs may then use the same AsyncSession only.
    """
    settings = get_settings()
    # TODO(retrieval-backend): dense_hits = await postgres_vector_search(session, query, ...)
    dense_hits = await qdrant.dense_search(query, limit=settings.retrieval_top_k)
    sparse_hits = await bm25_search(session, query, limit=settings.retrieval_top_k)
    candidates = merge_retrieval_hits(dense_hits, sparse_hits)
    return rerank_candidates(query, candidates, top_n=settings.rerank_top_n)
