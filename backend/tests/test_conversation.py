"""Multi-turn rewrite + provider parity tests for the retrieval query stage."""

from __future__ import annotations

import uuid

import pytest

from app.graph.conversation import ChatTurn, format_chat_history
from app.llm.grading import grade_retrieval
from app.llm.providers import RewrittenQuery, reset_llm_provider_cache
from app.llm.providers.heuristic import HeuristicProvider
from app.llm.retrieval_query import build_retrieval_query, build_rewritten_query
from app.retrieval.models import RetrievedChunk


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_llm_provider_cache()
    yield
    reset_llm_provider_cache()


@pytest.mark.asyncio
async def test_heuristic_rewrite_returns_latest_message_only():
    """Heuristic provider has no LLM, so follow-ups stay as the raw latest message."""
    prior = [
        ChatTurn(role="user", content="when do we need to use MFA?"),
        ChatTurn(
            role="assistant",
            content="MFA is required for production systems per the security policy.",
        ),
    ]
    query = await build_retrieval_query(prior, "so is it mandatory?")
    assert query == "so is it mandatory?"


@pytest.mark.asyncio
async def test_provider_rewrite_carries_subject_forward(monkeypatch: pytest.MonkeyPatch):
    """Any provider that rewrites should let the search query reference the prior topic."""
    prior = [
        ChatTurn(role="user", content="when do we need to use MFA?"),
        ChatTurn(role="assistant", content="MFA is required for production systems."),
    ]

    class _Stub(HeuristicProvider):
        name = "ollama"

        async def rewrite_query(self, history, latest):  # type: ignore[override]
            return RewrittenQuery(
                search_query="MFA mandatory production systems security policy",
                references_prior_turn=True,
            )

    monkeypatch.setattr("app.llm.providers.get_llm_provider", lambda: _Stub())
    rewritten = await build_rewritten_query(prior, "so is it mandatory?")
    assert "mfa" in rewritten.search_query.lower()
    assert rewritten.references_prior_turn is True


@pytest.mark.asyncio
async def test_provider_rewrite_can_switch_topic(monkeypatch: pytest.MonkeyPatch):
    """A topic-change follow-up must not carry forward old keywords."""
    prior = [
        ChatTurn(role="user", content="when do we need to use MFA?"),
        ChatTurn(role="assistant", content="MFA is required for production systems."),
    ]

    class _Stub(HeuristicProvider):
        name = "ollama"

        async def rewrite_query(self, history, latest):  # type: ignore[override]
            return RewrittenQuery(
                search_query="paid time off PTO accrual policy employees",
                references_prior_turn=False,
            )

    monkeypatch.setattr("app.llm.providers.get_llm_provider", lambda: _Stub())
    query = await build_retrieval_query(prior, "what about the PTOs?")
    assert "pto" in query.lower()
    assert "mfa" not in query.lower()


def test_format_chat_history_includes_roles():
    text = format_chat_history(
        [ChatTurn(role="user", content="Hello"), ChatTurn(role="assistant", content="Hi")]
    )
    assert "User: Hello" in text
    assert "Assistant: Hi" in text


@pytest.mark.asyncio
async def test_grade_passes_follow_up_when_retrieval_query_carries_subject():
    """Score-based grade must use the standalone retrieval query, not the raw follow-up."""
    security_chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=(
            "Access to production systems requires MFA and approval from the security team."
        ),
        score=0.5,
        relevance_score=0.363,
    )
    grade = await grade_retrieval(
        [security_chunk],
        "so is it mandatory?",
        retrieval_query="MFA mandatory production systems",
    )
    assert grade.should_abstain is False
