"""Provider protocols for the retrieval stack.

Mirrors ``app.llm.providers.base.LLMProvider``: thin, runtime-checkable
``Protocol`` interfaces selected by configuration, so the eval harness can swap
in-memory/offline implementations and the production path can use Qdrant +
hosted/local models without the calling code changing.

- ``Embedder``: text → dense vectors (and, later, sparse).
- ``Reranker``: (query, candidates) → reordered candidates with rerank scores.
- ``VectorStore``: dense (and later hybrid) nearest-neighbor over upserted chunks.
"""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

from app.retrieval.models import RetrievedChunk, VectorRecord


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors (async-safe)."""
        ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Reorder candidates and set their rerank score (async-safe)."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    name: str

    async def ensure_ready(self) -> None:
        """Create/verify backing storage (collection, index, etc.)."""
        ...

    async def upsert_chunks(
        self,
        document_id: uuid.UUID,
        chunk_ids: list[uuid.UUID],
        texts: list[str],
        bodies: list[str],
    ) -> None:
        """Embed ``texts`` and store points keyed by ``chunk_ids`` (clean ``bodies`` in payload).

        ``bodies`` is required (no fallback) — see Wave 0 PR4: an optional default
        silently re-opened the citation-prefix leak.
        """
        ...

    async def dense_search(self, query: str, limit: int = 20) -> list[VectorRecord]:
        """Return the top-``limit`` dense matches for ``query``."""
        ...
