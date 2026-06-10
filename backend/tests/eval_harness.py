"""In-memory retrieval stack used by the golden-set evaluation.

The full graph is exercised against this harness so the eval reflects real
behavior (chunking, dense + sparse fusion, rerank, grading, generation,
validation) without needing Postgres or Qdrant running.

The dense leg uses the project's real embedding stack (sentence-transformers)
and the sparse leg uses the same ``rank_bm25.BM25Okapi`` the production code
uses. There are no per-token rules, language-specific shortlists, or magic
thresholds baked in here — the eval gates on the same scoring semantics that
ship to production.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.ingestion.chunker import ChunkPiece, split_text_into_chunks
from app.retrieval.embeddings import embed_texts
from app.retrieval.models import RetrievedChunk
from app.retrieval.qdrant_store import VectorRecord
from app.retrieval.rerank import rerank_candidates_async


@dataclass
class _StoredChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    body: str
    embedded_text: str
    metadata: dict


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-9)


class InMemoryCorpus:
    """A corpus + retriever used by the golden eval.

    Behaviour mirrors :mod:`app.retrieval.hybrid`:

    - dense leg: sentence-transformers cosine over the embedded text (body +
      context prefix), same as production.
    - sparse leg: ``BM25Okapi`` over the same embedded text, same as
      production.

    Both legs feed RRF + rerank through the production code paths.
    """

    def __init__(self) -> None:
        self._chunks: list[_StoredChunk] = []
        self._embeddings: list[list[float]] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def ingest(self, *, filename: str, content: str) -> None:
        document_id = uuid.uuid4()
        pieces: list[ChunkPiece] = split_text_into_chunks(content, filename=filename)
        if not pieces:
            return
        new_embeddings = embed_texts([piece.content for piece in pieces])
        for piece, embedding in zip(pieces, new_embeddings, strict=True):
            stored = _StoredChunk(
                chunk_id=uuid.uuid4(),
                document_id=document_id,
                body=piece.body,
                embedded_text=piece.content,
                metadata={
                    **piece.metadata,
                    "context_prefix": piece.context_prefix,
                },
            )
            self._chunks.append(stored)
            self._embeddings.append(embedding)
            self._tokenized.append(stored.embedded_text.lower().split())
        self._bm25 = BM25Okapi(self._tokenized)

    def dense_search(self, query: str, *, limit: int) -> list[VectorRecord]:
        query_vec = embed_texts([query])[0]
        scored: list[tuple[_StoredChunk, float]] = [
            (chunk, _cosine(query_vec, embedding))
            for chunk, embedding in zip(self._chunks, self._embeddings, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            VectorRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.body,
                score=float(score),
            )
            for chunk, score in scored[:limit]
        ]

    def sparse_search(self, query: str, *, limit: int) -> list[RetrievedChunk]:
        if not self._chunks or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        scored = sorted(
            zip(self._chunks, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.body,
                score=float(score),
                metadata=chunk.metadata,
                sparse_score=float(score),
            )
            for chunk, score in scored[:limit]
            if float(score) > 0
        ]


async def hybrid_retrieve_in_memory(
    corpus: InMemoryCorpus,
    query: str,
    *,
    retrieval_top_k: int = 20,
    rerank_top_n: int = 5,
) -> list[RetrievedChunk]:
    """Mirror of :func:`app.retrieval.hybrid.hybrid_retrieve` against ``corpus``."""
    from app.retrieval.hybrid import merge_retrieval_hits

    dense_hits = corpus.dense_search(query, limit=retrieval_top_k)
    sparse_hits = corpus.sparse_search(query, limit=retrieval_top_k)
    candidates = merge_retrieval_hits(dense_hits, sparse_hits)
    return await rerank_candidates_async(query, candidates, top_n=rerank_top_n)
