from __future__ import annotations

import uuid

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.retrieval.embeddings import _DIM, embed_texts_async
from app.retrieval.models import VectorRecord

log = structlog.get_logger(__name__)


# TODO(retrieval-backend): Keep as Qdrant implementation of a shared VectorStore protocol.
# Postgres alternative: app/retrieval/postgres_vector_store.py (pgvector HNSW/IVFFlat).


class QdrantStore:
    name = "qdrant"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncQdrantClient(url=settings.qdrant_url)
        self._collection = settings.qdrant_collection

    async def ensure_ready(self) -> None:
        """VectorStore protocol alias for :meth:`ensure_collection`."""
        await self.ensure_collection()

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(size=_DIM, distance=qmodels.Distance.COSINE),
            )
            log.info("qdrant_collection_created", collection=self._collection)

    async def upsert_chunks(
        self,
        document_id: uuid.UUID,
        chunk_ids: list[uuid.UUID],
        texts: list[str],
        bodies: list[str],
    ) -> None:
        """Embed ``texts`` (prefixed) but store ``bodies`` (clean) in the payload.

        The dense vector is computed from the prefixed ``texts`` so retrieval
        still benefits from the structural context. The payload ``content`` is
        the clean body, so citation snippets never leak the prefix. Callers MUST
        pass the clean bodies; there is no implicit fallback, so the prefix can
        never silently leak into the payload.
        """
        await self.ensure_collection()
        vectors = await embed_texts_async(texts)
        points = [
            qmodels.PointStruct(
                id=str(chunk_id),
                vector=vector,
                payload={
                    "document_id": str(document_id),
                    "chunk_id": str(chunk_id),
                    "content": body,
                },
            )
            for chunk_id, vector, body in zip(
                chunk_ids, vectors, bodies, strict=True
            )
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def dense_search(self, query: str, limit: int = 20) -> list[VectorRecord]:
        await self.ensure_collection()
        query_vector = (await embed_texts_async([query]))[0]
        results = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        records: list[VectorRecord] = []
        for point in results.points:
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


def get_qdrant_store() -> QdrantStore:
    return QdrantStore()
