"""Cross-encoder rerank must time out gracefully and fall back to RRF order."""

from __future__ import annotations

import uuid

import pytest

from app.retrieval import rerank as rerank_module
from app.retrieval.models import RetrievedChunk


def _chunk(content: str, rrf: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        score=rrf,
        metadata={},
        rrf_score=rrf,
    )


@pytest.mark.asyncio
async def test_cross_encoder_timeout_falls_back_to_rrf_order(monkeypatch):
    """A reranker that exceeds rerank_timeout_s returns the RRF-ordered list."""
    candidates = [_chunk("alpha", 0.9), _chunk("bravo", 0.5), _chunk("charlie", 0.1)]

    class _HangingModel:
        def predict(self, pairs):  # noqa: ANN001
            import time

            # Cannot cancel a to_thread; sleep >> timeout so timing variance can't sneak a result back.  # noqa: E501
            time.sleep(2)
            return [0.0 for _ in pairs]

    monkeypatch.setattr(rerank_module, "_get_cross_encoder", lambda: _HangingModel())

    from app.core.config import get_settings

    monkeypatch.setenv("RERANK_TIMEOUT_S", "0.05")
    get_settings.cache_clear()

    result = await rerank_module.cross_encoder_rerank_async("query", candidates, top_n=2)

    get_settings.cache_clear()

    assert [chunk.content for chunk in result] == ["alpha", "bravo"]
    assert result[0].rerank_score is None
    assert result[0].ranking_score() == pytest.approx(0.9)
