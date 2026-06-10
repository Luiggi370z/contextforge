"""Multi-turn retrieval query rewrite via the active LLMProvider."""

from __future__ import annotations

import structlog

from app.graph.conversation import ChatTurn
from app.llm.providers import RewrittenQuery

log = structlog.get_logger(__name__)


async def build_retrieval_query(prior_turns: list[ChatTurn], message: str) -> str:
    """Return the standalone search query used by hybrid retrieval.

    Wraps :meth:`LLMProvider.rewrite_query` so callers that only need the search
    string (e.g. the graph runner) do not have to unpack the structured result.
    """
    rewritten = await build_rewritten_query(prior_turns, message)
    return rewritten.search_query


async def build_rewritten_query(
    prior_turns: list[ChatTurn], message: str
) -> RewrittenQuery:
    """Provider-agnostic structured rewrite (search_query + references_prior_turn)."""
    from app.llm.providers import get_llm_provider

    return await get_llm_provider().rewrite_query(prior_turns, message)
