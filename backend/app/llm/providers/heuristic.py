"""Deterministic, dependency-free provider used for tests and offline demos."""

from __future__ import annotations

import re

import structlog

from app.core.constants import (
    ABSTAIN_MESSAGE,
    DIRECT_GREETING_RESPONSE,
    ROUTE_DIRECT,
    ROUTE_MULTI_HOP,
    ROUTE_SINGLE_HOP_RAG,
)
from app.graph.conversation import ChatTurn
from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision, RouteKind
from app.llm.providers.base import RewrittenQuery
from app.retrieval.models import RetrievedChunk

log = structlog.get_logger(__name__)

_MULTI_HOP_KEYWORDS = ("step", "first", "then", "compare", "difference")
_GREETING_PATTERN = re.compile(r"^(hi|hello|hey)\b")
_GREETING_MAX_LEN = 12
_TOKEN_MIN_LEN = 5
_VALIDATE_TOP_TOKENS = 8
_VALIDATE_HIT_RATIO = 3
_HEURISTIC_ANSWER_WORD_LIMIT = 80


def heuristic_route(message: str) -> RouteDecision:
    """Cheap keyword/length classifier used as a baseline and a greeting fast-path."""
    lower = message.lower().strip()
    if _GREETING_PATTERN.match(lower) or len(lower) < _GREETING_MAX_LEN:
        return RouteDecision(route=ROUTE_DIRECT, confidence=0.9, reasoning="greeting or short")
    if any(keyword in lower for keyword in _MULTI_HOP_KEYWORDS):
        return RouteDecision(
            route=ROUTE_MULTI_HOP,
            confidence=0.75,
            reasoning="multi-part question keywords",
        )
    return RouteDecision(
        route=ROUTE_SINGLE_HOP_RAG,
        confidence=0.85,
        reasoning="default rag",
    )


def heuristic_validate(answer: str, contexts: list[str]) -> AnswerValidation:
    """Token-overlap check; kept as a safety net for the heuristic provider only."""
    if answer.strip() == ABSTAIN_MESSAGE:
        return AnswerValidation(grounded=True, issues=[])
    if not contexts:
        return AnswerValidation(grounded=False, issues=["no context"])
    joined = " ".join(contexts).lower()
    tokens = [token for token in answer.lower().split() if len(token) > _TOKEN_MIN_LEN]
    if not tokens:
        return AnswerValidation(grounded=True, issues=[])
    head = tokens[:_VALIDATE_TOP_TOKENS]
    hits = sum(1 for token in head if token in joined)
    grounded = hits >= max(1, len(head) // _VALIDATE_HIT_RATIO)
    issues = [] if grounded else ["answer not supported by retrieved context"]
    return AnswerValidation(grounded=grounded, issues=issues)


def heuristic_generate(
    *,
    query: str,
    contexts: list[str],
    route: RouteKind,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Template-only answer; the real generate path lives in the LLM providers."""
    if route == ROUTE_DIRECT:
        return DIRECT_GREETING_RESPONSE
    if not contexts:
        return ABSTAIN_MESSAGE
    numbered = "\n\n".join(
        f"[{index + 1}] {content}" for index, content in enumerate(contexts)
    )
    history_block = ""
    if chat_history and len(chat_history) > 1:
        prior_lines = [
            f"{turn['role'].capitalize()}: {turn['content'].strip()}"
            for turn in chat_history[:-1]
        ]
        history_block = "Conversation so far:\n" + "\n\n".join(prior_lines) + "\n\n"
    first_words = " ".join(contexts[0].split()[:_HEURISTIC_ANSWER_WORD_LIMIT])
    suffix = "..." if len(contexts[0]) > _HEURISTIC_ANSWER_WORD_LIMIT else ""
    return (
        f"{history_block}Based on the retrieved documents:\n\n{numbered}\n\n"
        f"Answer to your question ({query!r}): {first_words}{suffix}"
    )


class HeuristicProvider:
    """Pure-Python provider: deterministic, no network, no model weights."""

    name = "heuristic"

    async def rewrite_query(
        self, history: list[ChatTurn], latest: str
    ) -> RewrittenQuery:
        return RewrittenQuery(
            search_query=latest.strip(),
            references_prior_turn=False,
        )

    async def route(self, message: str) -> RouteDecision:
        return heuristic_route(message)

    async def grade(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        threshold: float,
        conversation: str | None = None,
    ) -> RetrievalGrade:
        from app.llm.grading import score_based_grade_retrieval

        return score_based_grade_retrieval(chunks)

    async def generate(
        self,
        *,
        query: str,
        contexts: list[str],
        route: RouteKind,
        chat_history: list[dict[str, str]] | None = None,
        retrieval_query: str | None = None,
    ) -> str:
        return heuristic_generate(
            query=query,
            contexts=contexts,
            route=route,
            chat_history=chat_history,
        )

    async def validate(
        self,
        *,
        answer: str,
        contexts: list[str],
    ) -> AnswerValidation:
        return heuristic_validate(answer, contexts)
