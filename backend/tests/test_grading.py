"""Tests for score-based grading + the LLM-judge fallback bridge."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.llm import grading
from app.llm.grading import (
    grade_retrieval,
    score_based_grade_retrieval,
    select_chunks_for_generation,
)
from app.llm.models import RetrievalGrade
from app.llm.providers.heuristic import HeuristicProvider
from app.retrieval.models import RetrievedChunk


def _settings(**overrides):
    base = dict(
        rerank_backend="lexical",
        grade_min_score=0.25,
        grade_min_score_cross_encoder=0.0,
        llm_provider="heuristic",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_score_grade_abstains_when_all_chunks_below_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(grading, "get_settings", lambda: _settings())
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
    monkeypatch.setattr(grading, "get_settings", lambda: _settings())
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="Access to production systems requires MFA.",
            score=0.72,
            relevance_score=0.72,
        )
    ]
    grade = score_based_grade_retrieval(chunks)
    assert grade.should_abstain is False


def test_score_grade_uses_top_rerank_score(monkeypatch: pytest.MonkeyPatch):
    """Whichever chunk has the highest post-rerank score sets the grade outcome."""
    monkeypatch.setattr(grading, "get_settings", lambda: _settings())
    pto_chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Employees accrue 20 PTO days per year.",
        score=0.9,
        relevance_score=0.05,
    )
    off_topic = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Random unrelated paragraph.",
        score=0.1,
        relevance_score=0.0,
    )
    grade = score_based_grade_retrieval([off_topic, pto_chunk])
    assert grade.should_abstain is False
    assert grade.score == pytest.approx(0.9)


def test_select_chunks_uses_rerank_order_and_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(grading, "get_settings", lambda: _settings())
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
    monkeypatch.setattr(grading, "get_settings", lambda: _settings())
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
    """If score-based grade does not abstain, the LLM judge must NOT be invoked."""

    class _Spy(HeuristicProvider):
        called = False

        async def grade(self, **_kwargs):  # type: ignore[override]
            _Spy.called = True
            return RetrievalGrade(relevant=False, score=0.0, should_abstain=True)

    monkeypatch.setattr(grading, "get_settings", lambda: _settings(llm_provider="ollama"))
    monkeypatch.setattr(
        "app.llm.providers.get_llm_provider", lambda: _Spy()
    )
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO accrual policy.",
            score=0.82,
            relevance_score=0.82,
        )
    ]
    grade = await grade_retrieval(chunks, "How many PTO days?")
    assert grade.should_abstain is False
    assert _Spy.called is False


@pytest.mark.asyncio
async def test_grade_consults_llm_judge_when_scores_abstain(monkeypatch: pytest.MonkeyPatch):
    """When score-based grade abstains and provider is non-heuristic, judge fires."""

    class _Judge(HeuristicProvider):
        name = "ollama"

        async def grade(self, **_kwargs):  # type: ignore[override]
            return RetrievalGrade(relevant=True, score=0.6, should_abstain=False)

    monkeypatch.setattr(grading, "get_settings", lambda: _settings(llm_provider="ollama"))
    monkeypatch.setattr(
        "app.llm.providers.get_llm_provider", lambda: _Judge()
    )
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="Production access requires MFA.",
            score=0.05,
            relevance_score=0.4,
        )
    ]
    grade = await grade_retrieval(chunks, "is it mandatory?", retrieval_query="MFA mandatory")
    assert grade.should_abstain is False
