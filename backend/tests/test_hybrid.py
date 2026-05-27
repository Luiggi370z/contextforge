import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.retrieval.hybrid import (
    bm25_search,
    hybrid_retrieve,
    merge_retrieval_hits,
    reciprocal_rank_fusion,
)
from app.retrieval.models import RetrievedChunk
from app.retrieval.qdrant_store import VectorRecord


def test_rrf_single_list_preserves_order():
    fused = reciprocal_rank_fusion([["alpha", "beta", "gamma"]])
    assert [chunk_id for chunk_id, _ in fused] == ["alpha", "beta", "gamma"]


def test_merge_retrieval_hits_boosts_chunks_in_both_lists():
    shared_id = uuid.uuid4()
    document_id = uuid.uuid4()
    dense = [
        VectorRecord(
            chunk_id=shared_id,
            document_id=document_id,
            content="PTO policy",
            score=0.4,
        )
    ]
    sparse = [
        RetrievedChunk(
            chunk_id=shared_id,
            document_id=document_id,
            content="PTO policy",
            score=0.2,
        )
    ]
    merged = merge_retrieval_hits(dense, sparse)
    assert len(merged) == 1
    assert merged[0].chunk_id == shared_id
    assert merged[0].score > 0


@pytest.mark.asyncio
async def test_bm25_search_ranks_relevant_chunk():
    chunk_pto = MagicMock()
    chunk_pto.id = uuid.uuid4()
    chunk_pto.document_id = uuid.uuid4()
    chunk_pto.content = "Employees receive paid time off PTO days annually."

    chunk_security = MagicMock()
    chunk_security.id = uuid.uuid4()
    chunk_security.document_id = uuid.uuid4()
    chunk_security.content = "Rotate passwords every ninety days for security."

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [chunk_security, chunk_pto]
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)

    hits = await bm25_search(session, "PTO days", limit=5)
    assert hits[0].chunk_id == chunk_pto.id


@pytest.mark.asyncio
async def test_hybrid_retrieve_combines_dense_and_sparse(monkeypatch):
    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    dense_hit = VectorRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content="remote work policy",
        score=0.8,
    )
    sparse_hit = RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content="remote work policy",
        score=0.6,
    )

    qdrant = AsyncMock()
    qdrant.dense_search = AsyncMock(return_value=[dense_hit])

    session = AsyncMock()
    monkeypatch.setattr(
        "app.retrieval.hybrid.bm25_search",
        AsyncMock(return_value=[sparse_hit]),
    )

    results = await hybrid_retrieve(session, qdrant, "remote work")
    assert len(results) >= 1
    assert results[0].content == "remote work policy"
