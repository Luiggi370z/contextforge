"""Standalone retrieval query for multi-turn chat (LLM condensation, not lexical rules)."""

from __future__ import annotations

import structlog

from app.core.config import get_settings
from app.core.constants import CONVERSATION_RETRIEVAL_QUERY_MAX_CHARS
from app.graph.conversation import ChatTurn, format_chat_history
from app.llm.ollama_provider import condense_retrieval_query as ollama_condense_retrieval_query

log = structlog.get_logger(__name__)


async def build_retrieval_query(prior_turns: list[ChatTurn], message: str) -> str:
    """Build the query string used for hybrid search.

    With an LLM provider, prior turns are condensed into one search query (standard query rewriting).
    In heuristic mode there is no rewriter—only the latest user message is searched. Short follow-ups
    in that mode rely on the architecture TODO (cross-encoder judge + provider-backed condense).
    """
    message = message.strip()
    if not prior_turns:
        return message

    settings = get_settings()
    if settings.llm_provider == "ollama":
        try:
            conversation = format_chat_history(prior_turns)
            condensed = await ollama_condense_retrieval_query(
                conversation=conversation,
                latest_message=message,
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )
            if condensed:
                return condensed[:CONVERSATION_RETRIEVAL_QUERY_MAX_CHARS]
        except Exception as exc:
            log.warning("retrieval_query_condense_fallback", error=str(exc))

    if settings.llm_provider in ("openai", "pydantic_ai"):
        # TODO: Instructor/Pydantic AI condense when API key present (see docs/TODO.md).
        log.debug("retrieval_query_condense_not_implemented", provider=settings.llm_provider)

    return message
