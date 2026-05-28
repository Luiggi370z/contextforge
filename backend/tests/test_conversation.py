import uuid
from types import SimpleNamespace

import pytest

from app.graph.conversation import ChatTurn, format_chat_history
from app.llm import retrieval_query
from app.llm.structured import grade_retrieval
from app.retrieval.models import RetrievedChunk


@pytest.mark.asyncio
async def test_heuristic_retrieval_query_is_latest_message_only(
    monkeypatch: pytest.MonkeyPatch,
):
    prior = [
        ChatTurn(role="user", content="when do we need to use MFA?"),
        ChatTurn(
            role="assistant",
            content="MFA is required for production systems per the security policy.",
        ),
    ]
    monkeypatch.setattr(
        retrieval_query,
        "get_settings",
        lambda: SimpleNamespace(llm_provider="heuristic"),
    )
    query = await retrieval_query.build_retrieval_query(prior, "so is it mandatory?")
    assert query == "so is it mandatory?"


@pytest.mark.asyncio
async def test_ollama_condenses_follow_up_for_retrieval(monkeypatch: pytest.MonkeyPatch):
    prior = [
        ChatTurn(role="user", content="when do we need to use MFA?"),
        ChatTurn(role="assistant", content="MFA is required for production systems."),
    ]

    async def fake_condense(**_: object) -> str:
        return "MFA mandatory production systems security policy"

    monkeypatch.setattr(
        retrieval_query,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="ollama",
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
        ),
    )
    monkeypatch.setattr(retrieval_query, "ollama_condense_retrieval_query", fake_condense)

    query = await retrieval_query.build_retrieval_query(prior, "so is it mandatory?")
    assert "mfa" in query.lower()


@pytest.mark.asyncio
async def test_ollama_condense_can_switch_topic(monkeypatch: pytest.MonkeyPatch):
    prior = [
        ChatTurn(role="user", content="when do we need to use MFA?"),
        ChatTurn(role="assistant", content="MFA is required for production systems."),
    ]

    async def fake_condense(**_: object) -> str:
        return "paid time off PTO accrual policy employees"

    monkeypatch.setattr(
        retrieval_query,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="ollama",
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
        ),
    )
    monkeypatch.setattr(retrieval_query, "ollama_condense_retrieval_query", fake_condense)

    query = await retrieval_query.build_retrieval_query(prior, "what about the PTOs?")
    assert "pto" in query.lower()
    assert "mfa" not in query.lower()


def test_format_chat_history_includes_roles():
    text = format_chat_history(
        [ChatTurn(role="user", content="Hello"), ChatTurn(role="assistant", content="Hi")]
    )
    assert "User: Hello" in text
    assert "Assistant: Hi" in text


def test_grade_passes_follow_up_when_retrieval_query_carries_subject():
    security_chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=(
            "Access to production systems requires MFA and approval from the security team."
        ),
        score=0.1,
        relevance_score=0.363,
    )
    grade = grade_retrieval(
        [security_chunk],
        "so is it mandatory?",
        retrieval_query="MFA mandatory production systems",
    )
    assert grade.should_abstain is False
