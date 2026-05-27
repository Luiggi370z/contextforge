"""Structured LLM outputs via Instructor, Pydantic AI, or heuristics."""

from __future__ import annotations

import re

import structlog

from app.core.config import get_settings
from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision, RouteKind

log = structlog.get_logger(__name__)


def _heuristic_route(message: str) -> RouteDecision:
    lower = message.lower().strip()
    if re.match(r"^(hi|hello|hey)\b", lower) or len(lower) < 12:
        return RouteDecision(route="direct", confidence=0.9, reasoning="greeting or short")
    if any(w in lower for w in ("step", "first", "then", "compare", "difference")):
        return RouteDecision(
            route="multi_hop",
            confidence=0.75,
            reasoning="multi-part question keywords",
        )
    return RouteDecision(route="single_hop_rag", confidence=0.85, reasoning="default rag")


async def decide_route(message: str) -> RouteDecision:
    settings = get_settings()
    if settings.llm_provider == "heuristic" or not settings.openai_api_key:
        return _heuristic_route(message)

    try:
        import instructor
        from openai import AsyncOpenAI

        client = instructor.from_openai(AsyncOpenAI(api_key=settings.openai_api_key))
        return await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user query route: direct (no docs), "
                        "single_hop_rag, or multi_hop."
                    ),
                },
                {"role": "user", "content": message},
            ],
            response_model=RouteDecision,
        )
    except Exception as exc:
        log.warning("route_llm_fallback", error=str(exc))
        return _heuristic_route(message)


def grade_retrieval(chunks: list, query: str) -> RetrievalGrade:
    settings = get_settings()
    if not chunks:
        return RetrievalGrade(relevant=False, score=0.0, should_abstain=True)
    top = max(c.score for c in chunks)
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
    tokens = [t for t in answer.lower().split() if len(t) > 5]
    if not tokens:
        return AnswerValidation(grounded=True, issues=[])
    hits = sum(1 for t in tokens[:8] if t in joined)
    grounded = hits >= max(1, len(tokens[:8]) // 3)
    issues = [] if grounded else ["answer not supported by retrieved context"]
    return AnswerValidation(grounded=grounded, issues=issues)


def generate_from_context(query: str, contexts: list[str], route: RouteKind) -> str:
    if route == "direct":
        return (
            "Hello! I am ContextForge. Upload policy documents and ask questions "
            "about them — I will cite sources from your corpus."
        )
    if not contexts:
        return "I don't have enough information in the indexed documents to answer that."
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        f"Based on the retrieved documents:\n\n{numbered}\n\n"
        f"Answer to your question ({query!r}): "
        + " ".join(contexts[0].split()[:80])
        + ("..." if len(contexts[0]) > 80 else "")
    )
