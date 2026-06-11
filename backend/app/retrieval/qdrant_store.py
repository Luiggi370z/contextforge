from __future__ import annotations

import asyncio
import uuid

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.retrieval.embeddings import embed_texts_async, embedding_dim
from app.retrieval.models import VectorRecord
from app.retrieval.sparse import embed_sparse_async

log = structlog.get_logger(__name__)


def _records_from_points(points) -> list[VectorRecord]:  # noqa: ANN001 — qdrant ScoredPoint
    """Map Qdrant scored points into ``VectorRecord``s, reading the chunk payload."""
    records: list[VectorRecord] = []
    for point in points:
        payload = point.payload or {}
        records.append(
            VectorRecord(
                chunk_id=uuid.UUID(str(payload["chunk_id"])),
                document_id=uuid.UUID(str(payload["document_id"])),
                content=str(payload.get("content", "")),
                score=float(point.score or 0.0),
            )
        )
    return records


# TODO(retrieval-backend): Keep as Qdrant implementation of a shared VectorStore protocol.
# Postgres alternative: app/retrieval/postgres_vector_store.py (pgvector HNSW/IVFFlat).


class QdrantStore:
    name = "qdrant"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncQdrantClient(url=settings.qdrant_url)
        # Collection name carries the dim so different embedders never share a
        # collection with mismatched vector sizes (switch ⇒ re-seed).
        self._collection = f"{settings.qdrant_collection}_{settings.embedding_dim}"

    async def ensure_ready(self) -> None:
        """VectorStore protocol alias for :meth:`ensure_collection`."""
        await self.ensure_collection()

    async def ensure_collection(self) -> None:
        settings = get_settings()
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    settings.dense_vector_name: qmodels.VectorParams(
                        size=embedding_dim(), distance=qmodels.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    settings.sparse_vector_name: qmodels.SparseVectorParams()
                },
            )
            log.info("qdrant_collection_created", collection=self._collection, hybrid=True)

    async def upsert_chunks(
        self,
        document_id: uuid.UUID,
        chunk_ids: list[uuid.UUID],
        texts: list[str],
        bodies: list[str],
    ) -> None:
        """Embed dense + sparse from ``texts``; store clean ``bodies`` in payload.

        ``bodies`` is required (no fallback) — see Wave 0 PR4: an optional default
        silently re-opened the citation-prefix leak.
        """
        await self.ensure_collection()
        settings = get_settings()
        dense_vectors, sparse_vectors = await asyncio.gather(
            embed_texts_async(texts), embed_sparse_async(texts)
        )
        points = [
            qmodels.PointStruct(
                id=str(chunk_id),
                vector={
                    settings.dense_vector_name: dense_vector,
                    settings.sparse_vector_name: qmodels.SparseVector(
                        indices=sparse[0], values=sparse[1]
                    ),
                },
                payload={
                    "document_id": str(document_id),
                    "chunk_id": str(chunk_id),
                    "content": body,
                },
            )
            for chunk_id, dense_vector, sparse, body in zip(
                chunk_ids, dense_vectors, sparse_vectors, bodies, strict=True
            )
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def dense_search(self, query: str, limit: int = 20) -> list[VectorRecord]:
        await self.ensure_collection()
        settings = get_settings()
        query_vector = (await embed_texts_async([query]))[0]
        results = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using=settings.dense_vector_name,
            limit=limit,
            with_payload=True,
        )
        return _records_from_points(results.points)

    async def hybrid_search(self, query: str, limit: int = 20) -> list[VectorRecord]:
        """Server-side RRF over dense + sparse legs; fused score on each record.

        The fused score is RRF (returned as ``VectorRecord.score``). The hybrid
        layer maps it into ``rrf_score`` so ``ranking_score()`` stays auditable.
        """
        await self.ensure_collection()
        settings = get_settings()
        dense_batch, sparse_batch = await asyncio.gather(
            embed_texts_async([query]), embed_sparse_async([query])
        )
        dense_vec = dense_batch[0]
        sparse = sparse_batch[0]
        results = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                qmodels.Prefetch(
                    query=dense_vec, using=settings.dense_vector_name, limit=limit
                ),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(indices=sparse[0], values=sparse[1]),
                    using=settings.sparse_vector_name,
                    limit=limit,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return _records_from_points(results.points)


def get_qdrant_store() -> QdrantStore:
    return QdrantStore()
