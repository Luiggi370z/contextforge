"""Retrieval relevance grading: rerank scores first, optional LLM judge fallback."""

from __future__ import annotations

import structlog

from app.core.config import get_settings
from app.core.constants import (
    CITATION_SCORE_RELATIVE_MIN,
    MAX_GENERATION_CONTEXTS,
    RERANK_BACKEND_CROSS_ENCODER,
)
from app.graph.conversation import ChatTurn, format_chat_history
from app.llm.models import RetrievalGrade
from app.llm.ollama_provider import grade_retrieval as ollama_grade_retrieval
from app.retrieval.models import RetrievedChunk

log = structlog.get_logger(__name__)


def effective_grade_min_score() -> float:
    """Minimum rerank/relevance score for a chunk to count as viable evidence."""
    settings = get_settings()
    if settings.rerank_backend == RERANK_BACKEND_CROSS_ENCODER:
        return settings.grade_min_score_cross_encoder
    return settings.grade_min_score


def rerank_score(chunk: RetrievedChunk) -> float:
    """Score from the reranker step (use for ordering evidence, not pre-rerank dense scores)."""
    return chunk.score


def evidence_score(chunk: RetrievedChunk) -> float:
    """Score for abstain threshold (cross-encoder sets score; lexical may use relevance)."""
    if chunk.relevance_score is not None:
        return max(chunk.score, chunk.relevance_score)
    return chunk.score


def score_based_grade_retrieval(chunks: list[RetrievedChunk]) -> RetrievalGrade:
    """Abstain when no retrieved chunk meets the configured rerank score threshold."""
    if not chunks:
        return RetrievalGrade(relevant=False, score=0.0, should_abstain=True)

    threshold = effective_grade_min_score()
    viable = [chunk for chunk in chunks if evidence_score(chunk) >= threshold]
    best_chunk = max(viable or chunks, key=evidence_score)
    top_score = evidence_score(best_chunk)
    should_abstain = not viable
    return RetrievalGrade(
        relevant=not should_abstain,
        score=top_score,
        should_abstain=should_abstain,
    )


def select_chunks_for_generation(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Top reranked chunks for generate + citations (tight band around best rerank score)."""
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
    """Grade retrieved evidence using rerank scores; optional LLM judge if scores are weak."""
    score_grade = score_based_grade_retrieval(chunks)
    if not score_grade.should_abstain:
        return score_grade

    settings = get_settings()
    judge_query = retrieval_query or query
    conversation: str | None = None
    if chat_history and len(chat_history) > 1:
        prior = [ChatTurn(role=turn["role"], content=turn["content"]) for turn in chat_history[:-1]]
        conversation = format_chat_history(prior)

    if settings.llm_provider == "ollama":
        try:
            return await ollama_grade_retrieval(
                query=judge_query,
                chunks=chunks,
                threshold=effective_grade_min_score(),
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                conversation=conversation,
            )
        except Exception as exc:
            log.warning("grade_llm_fallback", error=str(exc))

    return score_grade
