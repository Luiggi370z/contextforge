"""Retrieval relevance grading: rerank scores first, provider LLM judge fallback."""

from __future__ import annotations

import structlog

from app.core.config import get_settings
from app.core.constants import (
    CITATION_SCORE_RELATIVE_MIN,
    MAX_GENERATION_CONTEXTS,
    RERANK_BACKEND_CROSS_ENCODER,
    STAGE_GRADE_JUDGE,
)
from app.graph.conversation import ChatTurn, format_chat_history
from app.graph.progress import emit_stage
from app.llm.models import RetrievalGrade
from app.retrieval.models import RetrievedChunk

log = structlog.get_logger(__name__)


def effective_grade_min_score() -> float:
    """Minimum rerank score for a chunk to count as viable evidence."""
    settings = get_settings()
    if settings.rerank_backend == RERANK_BACKEND_CROSS_ENCODER:
        return settings.grade_min_score_cross_encoder
    return settings.grade_min_score


def rerank_score(chunk: RetrievedChunk) -> float:
    """Post-rerank score; falls back through earlier stage scores via ranking_score()."""
    return chunk.ranking_score()


def score_based_grade_retrieval(chunks: list[RetrievedChunk]) -> RetrievalGrade:
    """Abstain when no chunk meets the configured rerank score threshold.

    The score is the single ordering signal (rerank → rrf → raw). When this
    abstains, ``grade_retrieval`` consults the configured LLM provider's judge
    as a second opinion before giving up. There are no per-token / per-language
    rules here on purpose — that work belongs in the embedder, the reranker,
    and the judge.
    """
    if not chunks:
        return RetrievalGrade(relevant=False, score=0.0, should_abstain=True)

    threshold = effective_grade_min_score()
    viable = [chunk for chunk in chunks if rerank_score(chunk) >= threshold]
    best_chunk = max(viable or chunks, key=rerank_score)
    top_score = rerank_score(best_chunk)
    should_abstain = not viable
    return RetrievalGrade(
        relevant=not should_abstain,
        score=top_score,
        should_abstain=should_abstain,
    )


def select_chunks_for_generation(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Top reranked chunks for generation + citations (tight band around top score)."""
    if not chunks:
        return []

    threshold = effective_grade_min_score()
    ranked = sorted(chunks, key=rerank_score, reverse=True)
    top_chunk = ranked[0]
    top_score = rerank_score(top_chunk)

    if top_score < threshold:
        return [top_chunk]

    selected = [top_chunk]
    for chunk in ranked[1:]:
        score = rerank_score(chunk)
        if score < threshold:
            break
        if score >= top_score * CITATION_SCORE_RELATIVE_MIN:
            selected.append(chunk)
        else:
            break

    return selected[:MAX_GENERATION_CONTEXTS]


async def grade_retrieval(
    chunks: list[RetrievedChunk],
    query: str,
    *,
    retrieval_query: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> RetrievalGrade:
    """Grade evidence using rerank scores; consult the LLM judge when scores abstain."""
    score_grade = score_based_grade_retrieval(chunks)
    if not score_grade.should_abstain:
        return score_grade

    from app.llm.providers import get_llm_provider

    judge_query = retrieval_query or query
    conversation: str | None = None
    if chat_history and len(chat_history) > 1:
        prior = [
            ChatTurn(role=turn["role"], content=turn["content"])
            for turn in chat_history[:-1]
        ]
        conversation = format_chat_history(prior)

    provider = get_llm_provider()
    if provider.name == "heuristic":
        return score_grade

    emit_stage(STAGE_GRADE_JUDGE, provider=provider.name)
    try:
        return await provider.grade(
            query=judge_query,
            chunks=chunks,
            threshold=effective_grade_min_score(),
            conversation=conversation,
        )
    except Exception as exc:
        log.warning("grade_llm_fallback", error=str(exc))
        return score_grade
