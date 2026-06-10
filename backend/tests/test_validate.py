"""validate_node must enforce ungrounded → abstain for every LLM provider."""

from __future__ import annotations

import pytest

from app.core.constants import ABSTAIN_MESSAGE
from app.graph.nodes import validate_node
from app.llm.models import AnswerValidation
from app.llm.providers import reset_llm_provider_cache


@pytest.fixture(autouse=True)
def _clear():
    reset_llm_provider_cache()
    yield
    reset_llm_provider_cache()


@pytest.mark.asyncio
async def test_validate_replaces_answer_when_provider_says_ungrounded(
    monkeypatch: pytest.MonkeyPatch,
):
    """If the provider's validator says not grounded, the node must abstain."""

    async def _ungrounded(answer: str, contexts):
        return AnswerValidation(grounded=False, issues=["unsupported claim"])

    monkeypatch.setattr("app.graph.nodes.validate_answer", _ungrounded)
    state = {
        "route": "single_hop_rag",
        "answer": "Some claim not in context.",
        "abstained": False,
        "_selected_chunks": [
            type("C", (), {"content": "Only PTO information here."})(),
        ],
        "nodes_visited": [],
    }
    result = await validate_node(state)  # type: ignore[arg-type]
    assert result["answer"] == ABSTAIN_MESSAGE
    assert result["abstained"] is True
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_validate_keeps_answer_when_grounded(monkeypatch: pytest.MonkeyPatch):
    """A grounded validation result must leave the answer (and citations) alone."""

    async def _grounded(answer: str, contexts):
        return AnswerValidation(grounded=True, issues=[])

    monkeypatch.setattr("app.graph.nodes.validate_answer", _grounded)
    state = {
        "route": "single_hop_rag",
        "answer": "MFA is required for production [1].",
        "abstained": False,
        "_selected_chunks": [
            type("C", (), {"content": "Access to production systems requires MFA."})(),
        ],
        "nodes_visited": [],
    }
    result = await validate_node(state)  # type: ignore[arg-type]
    assert "answer" not in result  # node leaves it untouched


@pytest.mark.asyncio
async def test_validate_skips_when_abstained():
    """Already-abstained answers must not be re-validated."""
    state = {
        "route": "single_hop_rag",
        "answer": ABSTAIN_MESSAGE,
        "abstained": True,
        "_selected_chunks": [],
        "nodes_visited": [],
    }
    result = await validate_node(state)  # type: ignore[arg-type]
    assert "answer" not in result
