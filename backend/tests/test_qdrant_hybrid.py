"""Qdrant-native hybrid: fused results map into rrf_score and rerank correctly."""

from __future__ import annotations

import uuid
from typing import cast

import pytest

from app.retrieval.models import RetrievedChunk
from app.retrieval.qdrant_store import QdrantStore, VectorRecord


@pytest.mark.asyncio
async def test_hybrid_retrieve_maps_fused_score_to_rrf(monkeypatch):
    """hybrid_retrieve should set rrf_score from the fused Qdrant score."""
    from app.retrieval import hybrid as hybrid_module

    cid = uuid.uuid4()
    did = uuid.uuid4()

    class _FakeStore:
        async def hybrid_search(self, query, limit=20):  # noqa: ANN001
            return [
                VectorRecord(chunk_id=cid, document_id=did, content="20 PTO days.", score=0.83)
            ]

    async def _fake_rerank(query, candidates, top_n):  # noqa: ANN001
        # identity rerank — just confirm rrf_score flowed through
        return candidates[:top_n]

    monkeypatch.setattr(hybrid_module, "rerank_candidates_async", _fake_rerank)

    result = await hybrid_module.hybrid_retrieve_native(
        cast(QdrantStore, _FakeStore()), "pto days"
    )
    assert len(result) == 1
    chunk = result[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.rrf_score == pytest.approx(0.83)
    assert chunk.content == "20 PTO days."
    assert not chunk.content.startswith("Document: ")
