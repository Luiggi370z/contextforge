"""Structured LLM outputs via Instructor, Pydantic AI, or heuristics."""

from __future__ import annotations

import re

import structlog

from app.core.config import get_settings
from app.core.constants import (
    ABSTAIN_MESSAGE,
    CONTEXT_SELECTION_MIN_OVERLAP,
    DIRECT_GREETING_RESPONSE,
    GRADE_MIN_QUERY_TERM_OVERLAP,
    MAX_GENERATION_CONTEXTS,
    ROUTE_DIRECT,
    ROUTE_MULTI_HOP,
    ROUTE_SINGLE_HOP_RAG,
)
from app.llm.query_terms import (
    discriminative_terms,
    grading_terms,
    query_term_overlap_ratio,
    term_overlap_ratio,
)
from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision, RouteKind
from app.llm.ollama_provider import (
    decide_route as ollama_decide_route,
)
from app.llm.ollama_provider import (
    generate_from_context as ollama_generate_from_context,
)
from app.retrieval.models import RetrievedChunk

log = structlog.get_logger(__name__)


def _heuristic_route(message: str) -> RouteDecision:
    lower = message.lower().strip()
    if re.match(r"^(hi|hello|hey)\b", lower) or len(lower) < 12:
        return RouteDecision(route=ROUTE_DIRECT, confidence=0.9, reasoning="greeting or short")
    multi_hop_keywords = ("step", "first", "then", "compare", "difference")
    if any(keyword in lower for keyword in multi_hop_keywords):
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


async def _decide_route_ollama(message: str) -> RouteDecision:
    settings = get_settings()
    return await ollama_decide_route(
        message,
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )


async def decide_route(message: str) -> RouteDecision:
    settings = get_settings()
    heuristic = _heuristic_route(message)
    if settings.llm_provider == "heuristic":
        return heuristic
    if settings.llm_provider == "ollama":
        # Local models often label policy questions as direct; trust heuristics for RAG paths.
        if heuristic.route != ROUTE_DIRECT:
            return heuristic
        try:
            return await _decide_route_ollama(message)
        except Exception as exc:
            log.warning("route_llm_fallback", error=str(exc))
            return heuristic
    if settings.llm_provider in ("openai", "pydantic_ai") and not settings.openai_api_key:
        log.warning("route_llm_missing_openai_key", provider=settings.llm_provider)
        return heuristic

    try:
        if settings.llm_provider == "pydantic_ai":
            return await _decide_route_pydantic_ai(message)
        return await _decide_route_instructor(message)
    except Exception as exc:
        log.warning("route_llm_fallback", error=str(exc))
        return heuristic


def select_contexts_for_generation(
    query: str,
    contexts: list[str],
    *,
    retrieval_query: str | None = None,
) -> list[str]:
    """Return top snippets for generation/citations; exclude weak tangential matches."""
    if not contexts:
        return contexts
    terms = grading_terms(query, retrieval_query)
    if not terms:
        return contexts[:MAX_GENERATION_CONTEXTS]

    selection_terms = discriminative_terms(terms, contexts)
    ranked = sorted(
        contexts,
        key=lambda ctx: term_overlap_ratio(selection_terms, ctx),
        reverse=True,
    )
    selected = [
        ctx
        for ctx in ranked
        if term_overlap_ratio(selection_terms, ctx) >= CONTEXT_SELECTION_MIN_OVERLAP
    ]
    if not selected:
        selected = [ranked[0]]
    return selected[:MAX_GENERATION_CONTEXTS]


def grade_retrieval(
    chunks: list[RetrievedChunk],
    query: str,
    *,
    retrieval_query: str | None = None,
) -> RetrievalGrade:
    # Grading uses deterministic overlap + scores; Ollama grade was too permissive on weak matches.
    return _heuristic_grade_retrieval(chunks, query, retrieval_query=retrieval_query)


def _chunk_is_viable_for_query(
    chunk: RetrievedChunk,
    query: str,
    *,
    min_score: float,
    retrieval_query: str | None = None,
) -> bool:
    overlap = query_term_overlap_ratio(query, chunk.content, retrieval_query=retrieval_query)
    return (
        chunk.grading_score() >= min_score
        and overlap >= GRADE_MIN_QUERY_TERM_OVERLAP
    )


def _heuristic_grade_retrieval(
    chunks: list[RetrievedChunk],
    query: str,
    *,
    retrieval_query: str | None = None,
) -> RetrievalGrade:
    settings = get_settings()
    if not chunks:
        return RetrievalGrade(relevant=False, score=0.0, should_abstain=True)

    terms = grading_terms(query, retrieval_query)
    viable = [
        chunk
        for chunk in chunks
        if _chunk_is_viable_for_query(
            chunk,
            query,
            min_score=settings.grade_min_score,
            retrieval_query=retrieval_query,
        )
    ]
    should_abstain = not viable
    best_chunk = max(
        viable or chunks,
        key=lambda chunk: (
            term_overlap_ratio(terms, chunk.content),
            chunk.grading_score(),
        ),
    )
    top_score = best_chunk.grading_score()
    return RetrievalGrade(
        relevant=not should_abstain,
        score=min(top_score, 1.0),
        should_abstain=should_abstain,
    )


def validate_answer(answer: str, contexts: list[str]) -> AnswerValidation:
    return _heuristic_validate_answer(answer, contexts)


def _heuristic_validate_answer(answer: str, contexts: list[str]) -> AnswerValidation:
    if answer.strip() == ABSTAIN_MESSAGE:
        return AnswerValidation(grounded=True, issues=[])
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


def generate_from_context(
    query: str,
    contexts: list[str],
    route: RouteKind,
    *,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    settings = get_settings()
    if settings.llm_provider == "ollama" and route != ROUTE_DIRECT and contexts:
        try:
            return ollama_generate_from_context(
                query=query,
                contexts=contexts,
                route=route,
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                chat_history=chat_history,
            )
        except Exception as exc:
            log.warning("generate_llm_fallback", error=str(exc))
            return _heuristic_generate_from_context(
                query, contexts, route, chat_history=chat_history
            )
    return _heuristic_generate_from_context(query, contexts, route, chat_history=chat_history)


def _heuristic_generate_from_context(
    query: str,
    contexts: list[str],
    route: RouteKind,
    *,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    if route == ROUTE_DIRECT:
        return DIRECT_GREETING_RESPONSE
    if not contexts:
        return ABSTAIN_MESSAGE
    numbered = "\n\n".join(f"[{index + 1}] {content}" for index, content in enumerate(contexts))
    history_block = ""
    if chat_history and len(chat_history) > 1:
        prior_lines = [
            f"{turn['role'].capitalize()}: {turn['content'].strip()}"
            for turn in chat_history[:-1]
        ]
        history_block = f"Conversation so far:\n" + "\n\n".join(prior_lines) + "\n\n"
    return (
        f"{history_block}Based on the retrieved documents:\n\n{numbered}\n\n"
        f"Answer to your question ({query!r}): "
        + " ".join(contexts[0].split()[:80])
        + ("..." if len(contexts[0]) > 80 else "")
    )
