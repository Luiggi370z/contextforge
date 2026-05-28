import uuid
from types import SimpleNamespace

import pytest

from app.llm import structured
from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision
from app.llm.structured import _heuristic_route, grade_retrieval
from app.retrieval.models import RetrievedChunk


def test_heuristic_route_direct():
    d = _heuristic_route("hi")
    assert d.route == "direct"


def test_heuristic_route_rag():
    d = _heuristic_route("How many PTO days do employees get per year?")
    assert d.route == "single_hop_rag"


def test_heuristic_route_multi_hop():
    d = _heuristic_route("First compare PTO then explain remote work steps")
    assert d.route == "multi_hop"


def test_grade_abstain_on_empty():
    g = grade_retrieval([], "query")
    assert g.should_abstain is True


def test_grade_passes_with_chunks():
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO policy text",
            score=0.9,
        )
    ]
    g = grade_retrieval(chunks, "PTO")
    assert g.should_abstain is False


@pytest.mark.asyncio
async def test_decide_route_ollama_provider_path(monkeypatch: pytest.MonkeyPatch):
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

    async def fake_decide_route_ollama(message: str) -> RouteDecision:
        assert message == "Route this"
        return RouteDecision(route="single_hop_rag", confidence=0.9, reasoning="ollama")

    monkeypatch.setattr(structured, "_decide_route_ollama", fake_decide_route_ollama)
    decision = await structured.decide_route("Route this")
    assert decision.route == "single_hop_rag"


@pytest.mark.asyncio
async def test_decide_route_ollama_fallback_to_heuristic(monkeypatch: pytest.MonkeyPatch):
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

    async def failing_decide_route_ollama(_: str) -> RouteDecision:
        raise RuntimeError("ollama down")

    monkeypatch.setattr(structured, "_decide_route_ollama", failing_decide_route_ollama)
    decision = await structured.decide_route("How does PTO accrual work?")
    assert decision.route == "single_hop_rag"


def test_generate_from_context_ollama_provider_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        structured,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="ollama",
            grade_min_score=0.25,
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
        ),
    )

    def fake_generate_from_context(**kwargs: object) -> str:
        assert kwargs["query"] == "What is PTO?"
        return "Ollama answer"

    monkeypatch.setattr(structured, "ollama_generate_from_context", fake_generate_from_context)
    answer = structured.generate_from_context(
        "What is PTO?", ["PTO policy context"], "single_hop_rag"
    )
    assert answer == "Ollama answer"


def test_generate_from_context_ollama_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        structured,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="ollama",
            grade_min_score=0.25,
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
        ),
    )

    def failing_generate_from_context(**_: object) -> str:
        raise RuntimeError("bad response")

    monkeypatch.setattr(structured, "ollama_generate_from_context", failing_generate_from_context)
    answer = structured.generate_from_context(
        "What is PTO?", ["PTO policy context"], "single_hop_rag"
    )
    assert answer.startswith("Based on the retrieved documents:")


def test_grade_retrieval_ollama_provider_path(monkeypatch: pytest.MonkeyPatch):
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO policy text",
            score=0.9,
        )
    ]
    monkeypatch.setattr(
        structured,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="ollama",
            grade_min_score=0.25,
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
        ),
    )

    def fake_grade_retrieval(**kwargs: object) -> RetrievalGrade:
        assert kwargs["query"] == "PTO"
        return RetrievalGrade(relevant=True, score=0.95, should_abstain=False)

    monkeypatch.setattr(structured, "ollama_grade_retrieval", fake_grade_retrieval)
    grade = structured.grade_retrieval(chunks, "PTO")
    assert grade.should_abstain is False


def test_validate_answer_ollama_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        structured,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="ollama",
            grade_min_score=0.25,
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
        ),
    )

    def failing_validate_answer(**_: object) -> AnswerValidation:
        raise RuntimeError("malformed json")

    monkeypatch.setattr(structured, "ollama_validate_answer", failing_validate_answer)
    validation = structured.validate_answer("answer text", ["supporting context"])
    assert validation.grounded is False
