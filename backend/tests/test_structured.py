import uuid
from types import SimpleNamespace

import pytest

from app.llm import structured
from app.llm.grading import grade_retrieval, select_chunks_for_generation
from app.llm.models import RouteDecision
from app.llm.structured import _heuristic_route
from app.retrieval.models import RetrievedChunk


def test_heuristic_route_direct():
    decision = _heuristic_route("hi")
    assert decision.route == "direct"


def test_heuristic_route_rag():
    decision = _heuristic_route("How many PTO days do employees get per year?")
    assert decision.route == "single_hop_rag"


def test_heuristic_route_multi_hop():
    decision = _heuristic_route("First compare PTO then explain remote work steps")
    assert decision.route == "multi_hop"


@pytest.mark.asyncio
async def test_grade_abstain_on_empty():
    grade = await grade_retrieval([], "query")
    assert grade.should_abstain is True


@pytest.mark.asyncio
async def test_grade_passes_with_high_rerank_score(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        structured,
        "get_settings",
        lambda: SimpleNamespace(
            rerank_backend="lexical",
            grade_min_score=0.25,
            grade_min_score_cross_encoder=0.0,
            llm_provider="heuristic",
        ),
    )
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO policy text",
            score=0.1,
            relevance_score=0.9,
        )
    ]
    grade = await grade_retrieval(chunks, "PTO")
    assert grade.should_abstain is False


@pytest.mark.asyncio
async def test_grade_abstains_when_rerank_scores_low(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        structured,
        "get_settings",
        lambda: SimpleNamespace(
            rerank_backend="lexical",
            grade_min_score=0.25,
            grade_min_score_cross_encoder=0.0,
            llm_provider="heuristic",
        ),
    )
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO policy for full-time employees.",
            score=0.03,
            relevance_score=0.05,
        )
    ]
    grade = await grade_retrieval(chunks, "What is the lunar landing budget for 2099?")
    assert grade.should_abstain is True


@pytest.mark.asyncio
async def test_decide_route_ollama_uses_heuristic_for_rag_questions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        structured,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="ollama",
            openai_api_key=None,
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
        ),
    )

    async def fail_if_called(_: str) -> RouteDecision:
        raise AssertionError("ollama route should not run for policy questions")

    monkeypatch.setattr(structured, "_decide_route_ollama", fail_if_called)
    decision = await structured.decide_route("How many PTO days do employees get?")
    assert decision.route == "single_hop_rag"


def test_select_chunks_prefers_top_rerank_score(monkeypatch: pytest.MonkeyPatch):
    """Selection ranks by post-rerank score so weak-rerank tails are dropped from citations."""
    monkeypatch.setattr(
        structured,
        "get_settings",
        lambda: SimpleNamespace(
            rerank_backend="lexical",
            grade_min_score=0.25,
            grade_min_score_cross_encoder=0.0,
        ),
    )
    weak_rerank = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Off-topic but dense-similar passage.",
        score=0.2,
        relevance_score=0.8,
    )
    top_rerank = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Best reranked answer snippet.",
        score=0.5,
        relevance_score=0.05,
    )
    selected = select_chunks_for_generation([weak_rerank, top_rerank])
    assert len(selected) == 1
    assert "Best reranked" in selected[0].content
