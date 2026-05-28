import uuid
from types import SimpleNamespace

import pytest

from app.llm import structured
from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision
from app.llm.structured import _heuristic_route, grade_retrieval, select_contexts_for_generation
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


def test_grade_abstains_when_query_terms_missing_from_top_chunk():
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO policy for full-time employees.",
            score=0.032,
            relevance_score=0.82,
        )
    ]
    grade = grade_retrieval(chunks, "What is the lunar landing budget for 2099?")
    assert grade.should_abstain is True


def test_grade_passes_when_query_terms_match_top_chunk():
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="Compare PTO policy steps and remote work approval steps.",
            score=0.032,
            relevance_score=0.82,
        )
    ]
    grade = grade_retrieval(
        chunks,
        "Compare PTO policy steps with remote work approval steps",
    )
    assert grade.should_abstain is False


def test_grade_passes_mfa_question_on_security_policy_chunk():
    """Short acronyms like MFA must count toward overlap (len >= 3, not > 3)."""
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content=(
                "Access to production systems requires MFA and approval from the security team."
            ),
            score=0.108,
            relevance_score=0.72,
        )
    ]
    grade = grade_retrieval(chunks, "do we need to use MFA always?")
    assert grade.should_abstain is False


def test_grade_passes_when_matching_chunk_is_not_highest_scored():
    pto_chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Employees receive fifteen days of PTO per year.",
        score=0.9,
        relevance_score=0.886,
    )
    security_chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=(
            "Access to production systems requires MFA and approval from the security team."
        ),
        score=0.1,
        relevance_score=0.403,
    )
    grade = grade_retrieval(
        [pto_chunk, security_chunk],
        "when do we need to use MFA?",
    )
    assert grade.should_abstain is False


def test_select_contexts_prefers_matching_snippets():
    contexts = [
        "Employees receive fifteen days of PTO per year.",
        "Access to production systems requires MFA and approval from the security team.",
    ]
    selected = select_contexts_for_generation("when do we need to use MFA?", contexts)
    assert selected[0].startswith("Access to production")


def test_select_contexts_includes_pto_chunk_for_pto_question():
    pto = "Full-time employees accrue 20 PTO days per calendar year."
    security = (
        "Access to production systems requires MFA and approval from the security team."
    )
    selected = select_contexts_for_generation(
        "what about the PTOs?",
        [security, pto],
        retrieval_query="what about the PTOs?",
    )
    assert len(selected) == 1
    assert "PTO" in selected[0]


def test_select_contexts_excludes_tangential_policy_on_follow_up():
    remote_work = (
        "Employees may work remotely up to three days per week with manager approval. "
        "All remote workers must use company-approved VPN for accessing internal systems."
    )
    security = (
        "Access to production systems requires MFA and approval from the security team."
    )
    selected = select_contexts_for_generation(
        "so is it mandatory?",
        [remote_work, security],
        retrieval_query="MFA mandatory production systems security",
    )
    assert len(selected) == 1
    assert "MFA" in selected[0]


def test_grade_passes_when_rrf_score_low_but_relevance_high():
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO policy text",
            score=0.032,
            relevance_score=0.82,
        )
    ]
    g = grade_retrieval(chunks, "PTO")
    assert g.should_abstain is False


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


def test_grade_retrieval_uses_heuristic_when_provider_is_ollama():
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO policy text for employees",
            score=0.032,
            relevance_score=0.82,
        )
    ]
    grade = grade_retrieval(chunks, "How many PTO days do employees get?")
    assert grade.should_abstain is False
