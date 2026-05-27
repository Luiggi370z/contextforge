"""Structured LLM outputs via Instructor, Pydantic AI, or heuristics."""

from __future__ import annotations

import re

import structlog

from app.core.config import get_settings
from app.core.constants import (
    ABSTAIN_MESSAGE,
    DIRECT_GREETING_RESPONSE,
    ROUTE_DIRECT,
    ROUTE_MULTI_HOP,
    ROUTE_SINGLE_HOP_RAG,
)
from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision, RouteKind
from app.retrieval.models import RetrievedChunk

log = structlog.get_logger(__name__)


def _heuristic_route(message: str) -> RouteDecision:
    lower = message.lower().strip()
    if re.match(r"^(hi|hello|hey)\b", lower) or len(lower) < 12:
        return RouteDecision(route=ROUTE_DIRECT, confidence=0.9, reasoning="greeting or short")
    if any(w in lower for w in ("step", "first", "then", "compare", "difference")):
        return RouteDecision(
            route=ROUTE_MULTI_HOP,
            confidence=0.75,
            reasoning="multi-part question keywords",
        )
    return RouteDecision(
        route=ROUTE_SINGLE_HOP_RAG, confidence=0.85, reasoning="default rag"
    )


async def _decide_route_instructor(message: str) -> RouteDecision:
    from app.llm.instructor_route import decide_route

    return await decide_route(message)


async def _decide_route_pydantic_ai(message: str) -> RouteDecision:
    from app.llm.pydantic_ai_route import decide_route

    return await decide_route(message)


async def decide_route(message: str) -> RouteDecision:
    settings = get_settings()
    if settings.llm_provider == "heuristic" or not settings.openai_api_key:
        return _heuristic_route(message)

    try:
        if settings.llm_provider == "pydantic_ai":
            return await _decide_route_pydantic_ai(message)
        return await _decide_route_instructor(message)
    except Exception as exc:
        log.warning("route_llm_fallback", error=str(exc))
        return _heuristic_route(message)


def grade_retrieval(chunks: list[RetrievedChunk], query: str) -> RetrievalGrade:
    settings = get_settings()
    if not chunks:
        return RetrievalGrade(relevant=False, score=0.0, should_abstain=True)
    top = max(chunk.score for chunk in chunks)
    should_abstain = top < settings.grade_min_score
    return RetrievalGrade(
        relevant=not should_abstain,
        score=min(top, 1.0),
        should_abstain=should_abstain,
    )


def validate_answer(answer: str, contexts: list[str]) -> AnswerValidation:
    if not contexts:
        return AnswerValidation(grounded=False, issues=["no context"])
    joined = " ".join(contexts).lower()
    tokens = [token for token in answer.lower().split() if len(token) > 5]
    if not tokens:
        return AnswerValidation(grounded=True, issues=[])
    hits = sum(1 for token in tokens[:8] if token in joined)
    grounded = hits >= max(1, len(tokens[:8]) // 3)
    issues = [] if grounded else ["answer not supported by retrieved context"]
    return AnswerValidation(grounded=grounded, issues=issues)


def generate_from_context(query: str, contexts: list[str], route: RouteKind) -> str:
    if route == ROUTE_DIRECT:
        return DIRECT_GREETING_RESPONSE
    if not contexts:
        return ABSTAIN_MESSAGE
    numbered = "\n\n".join(f"[{index + 1}] {content}" for index, content in enumerate(contexts))
    return (
        f"Based on the retrieved documents:\n\n{numbered}\n\n"
        f"Answer to your question ({query!r}): "
        + " ".join(contexts[0].split()[:80])
        + ("..." if len(contexts[0]) > 80 else "")
    )
