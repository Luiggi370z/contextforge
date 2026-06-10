"""Hybrid dense + sparse retrieval with RRF fusion and provider-agnostic rerank."""

from __future__ import annotations

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.constants import RRF_RANK_CONSTANT
from app.db.models import Chunk
from app.retrieval.dedupe import dedupe_chunks_by_content
from app.retrieval.models import RetrievedChunk
from app.retrieval.qdrant_store import QdrantStore, VectorRecord
from app.retrieval.rerank import rerank_candidates_async


def reciprocal_rank_fusion(
    rank_lists: list[list[str]],
    rank_constant: int = RRF_RANK_CONSTANT,
) -> list[tuple[str, float]]:
    """Merge ranked id lists with Reciprocal Rank Fusion.

    Example:
        >>> reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0][0]
        'b'
    """
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank_constant + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def merge_retrieval_hits(
    dense_hits: list[VectorRecord],
    sparse_hits: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Fuse dense and sparse hits with RRF.

    Each chunk keeps its stage-local ``dense_score`` and ``sparse_score`` so
    later stages can audit where the evidence came from. ``rrf_score`` is the
    fused signal; ``score`` is set to ``rrf_score`` so unmodified consumers
    keep sorting by the most recent stage.
    """
    dense_ids = [str(hit.chunk_id) for hit in dense_hits]
    sparse_ids = [str(hit.chunk_id) for hit in sparse_hits]
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])

    dense_by_id = {str(hit.chunk_id): hit for hit in dense_hits}
    sparse_by_id = {str(hit.chunk_id): hit for hit in sparse_hits}

    candidates: list[RetrievedChunk] = []
    for chunk_id, rrf_score in fused:
        dense_hit = dense_by_id.get(chunk_id)
        sparse_hit = sparse_by_id.get(chunk_id)
        primary = dense_hit if dense_hit is not None else sparse_hit
        assert primary is not None  # one of the two must exist by construction
        # Sparse hits carry chunk metadata (filename, section, etc.) because the
        # DB row has it. Dense VectorRecord does not, so fall back to the sparse
        # side when we only have a dense hit's geometry.
        metadata = (
            getattr(sparse_hit, "metadata", None)
            or getattr(dense_hit, "metadata", None)
            or {}
        )
        candidates.append(
            RetrievedChunk(
                chunk_id=primary.chunk_id,
                document_id=primary.document_id,
                content=primary.content,
                score=rrf_score,
                metadata=metadata,
                dense_score=dense_hit.score if dense_hit is not None else None,
                sparse_score=sparse_hit.score if sparse_hit is not None else None,
                rrf_score=rrf_score,
                rerank_score=None,
            )
        )
    return candidates


# TODO(retrieval-backend): When RETRIEVAL_BACKEND=postgres, replace with Postgres FTS
# (tsvector + ts_rank / websearch_to_tsquery) instead of loading all chunks for BM25Okapi.


def _bm25_document(chunk: Chunk) -> str:
    """Sparse leg sees the same enriched text we embed (body + context prefix)."""
    metadata = chunk.metadata_ or {}
    prefix = metadata.get("context_prefix", "")
    if prefix:
        return f"{prefix}\n{chunk.content}"
    return chunk.content


async def bm25_search(
    session: AsyncSession, query: str, limit: int | None = None
) -> list[RetrievedChunk]:
    """Rank chunks in Postgres with BM25 over tokenized (body + context) content."""
    settings = get_settings()
    search_limit = limit if limit is not None else settings.retrieval_top_k
    result = await session.execute(select(Chunk))
    chunks = list(result.scalars().all())
    if not chunks:
        return []
    corpus = [_bm25_document(chunk) for chunk in chunks]
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
            metadata=chunk.metadata_ or {},
            sparse_score=float(score),
        )
        for chunk, score in ranked
    ]


async def hybrid_retrieve(
    session: AsyncSession,
    qdrant: QdrantStore,
    query: str,
) -> list[RetrievedChunk]:
    """Dense (Qdrant) + sparse (BM25) retrieval, RRF fusion, then rerank."""
    settings = get_settings()
    dense_hits = await qdrant.dense_search(query, limit=settings.retrieval_top_k)
    sparse_hits = await bm25_search(session, query, limit=settings.retrieval_top_k)
    candidates = merge_retrieval_hits(dense_hits, sparse_hits)
    ranked = await rerank_candidates_async(query, candidates, top_n=settings.rerank_top_n)
    return dedupe_chunks_by_content(ranked)
