import uuid
from types import SimpleNamespace

import pytest

from app.llm import grading
from app.llm.grading import (
    grade_retrieval,
    score_based_grade_retrieval,
    select_chunks_for_generation,
)
from app.retrieval.models import RetrievedChunk


def test_score_grade_abstains_when_all_chunks_below_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        grading,
        "get_settings",
        lambda: SimpleNamespace(
            rerank_backend="lexical",
            grade_min_score=0.25,
            grade_min_score_cross_encoder=0.0,
        ),
    )
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="Irrelevant policy text.",
            score=0.05,
            relevance_score=0.05,
        )
    ]
    grade = score_based_grade_retrieval(chunks)
    assert grade.should_abstain is True


def test_score_grade_passes_when_rerank_score_above_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        grading,
        "get_settings",
        lambda: SimpleNamespace(
            rerank_backend="lexical",
            grade_min_score=0.25,
            grade_min_score_cross_encoder=0.0,
        ),
    )
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="Access to production systems requires MFA.",
            score=0.1,
            relevance_score=0.72,
        )
    ]
    grade = score_based_grade_retrieval(chunks)
    assert grade.should_abstain is False


def test_score_grade_uses_viable_chunk_not_only_top_rrf(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        grading,
        "get_settings",
        lambda: SimpleNamespace(
            rerank_backend="lexical",
            grade_min_score=0.25,
            grade_min_score_cross_encoder=0.0,
        ),
    )
    pto_chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Employees accrue 20 PTO days per year.",
        score=0.9,
        relevance_score=0.05,
    )
    security_chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Access to production systems requires MFA.",
        score=0.1,
        relevance_score=0.72,
    )
    grade = score_based_grade_retrieval([pto_chunk, security_chunk])
    assert grade.should_abstain is False


def test_select_chunks_uses_rerank_order_and_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        grading,
        "get_settings",
        lambda: SimpleNamespace(
            rerank_backend="lexical",
            grade_min_score=0.25,
            grade_min_score_cross_encoder=0.0,
        ),
    )
    security = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="MFA required for production.",
        score=0.5,
        relevance_score=0.8,
    )
    remote = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Remote work VPN policy.",
        score=0.4,
        relevance_score=0.1,
    )
    selected = select_chunks_for_generation([security, remote])
    assert len(selected) == 1
    assert "MFA" in selected[0].content


def test_select_chunks_drops_high_dense_low_rerank_tail(monkeypatch: pytest.MonkeyPatch):
    """MFA-style case: dense relevance favors wrong doc; rerank score picks security only."""
    monkeypatch.setattr(
        grading,
        "get_settings",
        lambda: SimpleNamespace(
            rerank_backend="lexical",
            grade_min_score=0.25,
            grade_min_score_cross_encoder=0.0,
        ),
    )
    remote = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Remote work VPN policy.",
        score=0.35,
        relevance_score=0.886,
    )
    security = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Access to production systems requires MFA.",
        score=0.62,
        relevance_score=0.403,
    )
    selected = select_chunks_for_generation([remote, security])
    assert len(selected) == 1
    assert "MFA" in selected[0].content


@pytest.mark.asyncio
async def test_grade_skips_llm_when_rerank_scores_pass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        grading,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="ollama",
            rerank_backend="lexical",
            grade_min_score=0.25,
            grade_min_score_cross_encoder=0.0,
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
        ),
    )

    async def fail_llm(**_: object) -> None:
        raise AssertionError("LLM grade should not run when rerank scores pass")

    monkeypatch.setattr(grading, "ollama_grade_retrieval", fail_llm)
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO accrual policy.",
            score=0.1,
            relevance_score=0.82,
        )
    ]
    grade = await grade_retrieval(chunks, "How many PTO days?")
    assert grade.should_abstain is False
