"""Thin dispatcher over the active LLMProvider (route, generate, validate)."""

from __future__ import annotations

import structlog

from app.core.constants import STAGE_GENERATE_LLM
from app.graph.progress import emit_stage
from app.llm.grading import grade_retrieval, select_chunks_for_generation
from app.llm.models import AnswerValidation, RouteDecision, RouteKind
from app.llm.providers import get_llm_provider
from app.llm.providers.heuristic import heuristic_route

log = structlog.get_logger(__name__)

__all__ = [
    "decide_route",
    "generate_from_context",
    "grade_retrieval",
    "select_chunks_for_generation",
    "validate_answer",
    "heuristic_route",
]


async def decide_route(message: str) -> RouteDecision:
    """Classify a user message into a graph route via the active provider."""
    return await get_llm_provider().route(message)


async def generate_from_context(
    query: str,
    contexts: list[str],
    route: RouteKind,
    *,
    chat_history: list[dict[str, str]] | None = None,
    retrieval_query: str | None = None,
) -> str:
    """Run the active provider's grounded generation step.

    ``retrieval_query`` is the rewriter's resolved version of the user's question
    (subject carried forward for pronoun follow-ups). Providers pass it to the
    model as additional context so a turn like "so is it mandatory?" gets
    grounded against the MFA discussion that preceded it.
    """
    provider = get_llm_provider()
    emit_stage(STAGE_GENERATE_LLM, provider=provider.name)
    return await provider.generate(
        query=query,
        contexts=contexts,
        route=route,
        chat_history=chat_history,
        retrieval_query=retrieval_query,
    )


async def validate_answer(answer: str, contexts: list[str]) -> AnswerValidation:
    """Run the active provider's grounding/entailment check."""
    return await get_llm_provider().validate(answer=answer, contexts=contexts)
