"""Retrieval domain types.

The chunk carries scores from every stage of the pipeline so callers can
disambiguate (e.g. RRF score is for fusion ordering, rerank score is the
single ordering signal after rerank, dense/sparse are stage-local).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """A single retrieved chunk and every score we recorded for it.

    Stages populate scores additively:

    - ``dense_score``: cosine similarity from the vector store (Qdrant / pgvector).
    - ``sparse_score``: BM25 / FTS score from the sparse leg.
    - ``rrf_score``: Reciprocal Rank Fusion score after merging dense + sparse.
    - ``rerank_score``: post-rerank score (lexical blend or cross-encoder logit).

    ``score`` is the ordering signal for downstream stages: it equals
    ``rerank_score`` after the reranker runs, otherwise the most recent stage's score.
    Use :meth:`ranking_score` instead of reading ``score`` directly so the intent is clear.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float
    metadata: dict = field(default_factory=dict)
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    # Back-compat: deprecated; new code should not read this. Kept so we don't break
    # external consumers that touched the old single-blob ``relevance_score``. Phase
    # 7 cleanup may delete this entirely.
    relevance_score: float | None = None

    def ranking_score(self) -> float:
        """The single score downstream stages should sort/threshold against.

        Falls back through stages in order ``rerank_score → rrf_score → score``
        so the chunk is always comparable even before rerank runs.
        """
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        return self.score
