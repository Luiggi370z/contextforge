"""VectorStore implementations: Qdrant (production) + in-memory (eval/offline)."""

from __future__ import annotations

import math
import uuid

from app.retrieval.embeddings import _hash_embed, embed_texts
from app.retrieval.models import VectorRecord

__all__ = ["InMemoryVectorStore"]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-9)


class InMemoryVectorStore:
    """Dense store backed by a Python list — used by the eval gate (no Qdrant)."""

    name = "in_memory"

    def __init__(self, embedder_backend: str = "hash") -> None:
        self._embedder_backend = embedder_backend
        self._records: list[VectorRecord] = []
        self._vectors: list[list[float]] = []

    async def ensure_ready(self) -> None:
        return None

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder_backend == "hash":
            return [_hash_embed(text) for text in texts]
        import asyncio

        return await asyncio.to_thread(embed_texts, texts)

    async def upsert_chunks(
        self,
        document_id: uuid.UUID,
        chunk_ids: list[uuid.UUID],
        texts: list[str],
        bodies: list[str],
    ) -> None:
        vectors = await self._embed(texts)
        for chunk_id, vector, body in zip(chunk_ids, vectors, bodies, strict=True):
            self._records.append(
                VectorRecord(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=body,
                    score=0.0,
                )
            )
            self._vectors.append(vector)

    async def dense_search(self, query: str, limit: int = 20) -> list[VectorRecord]:
        if not self._records:
            return []
        query_vec = (await self._embed([query]))[0]
        scored = sorted(
            (
                VectorRecord(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    content=record.content,
                    score=_cosine(query_vec, vector),
                )
                for record, vector in zip(self._records, self._vectors, strict=True)
            ),
            key=lambda record: record.score,
            reverse=True,
        )
        return scored[:limit]
