"""Ollama-backed provider: local llama3.x via the Ollama chat API."""

from __future__ import annotations

import structlog

from app.core.constants import (
    CONVERSATION_RETRIEVAL_QUERY_MAX_CHARS,
    ROUTE_DIRECT,
)
from app.graph.conversation import ChatTurn, format_chat_history
from app.llm import ollama_provider
from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision, RouteKind
from app.llm.providers.base import RewrittenQuery
from app.llm.providers.heuristic import (
    HeuristicProvider,
    heuristic_generate,
    heuristic_route,
    heuristic_validate,
)
from app.retrieval.models import RetrievedChunk

log = structlog.get_logger(__name__)


class OllamaProvider:
    """LLMProvider backed by a local Ollama server."""

    name = "ollama"

    def __init__(self, *, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self._fallback = HeuristicProvider()

    async def rewrite_query(
        self, history: list[ChatTurn], latest: str
    ) -> RewrittenQuery:
        latest = latest.strip()
        if not history:
            return RewrittenQuery(search_query=latest, references_prior_turn=False)
        try:
            conversation = format_chat_history(history)
            condensed = await ollama_provider.condense_retrieval_query(
                conversation=conversation,
                latest_message=latest,
                base_url=self.base_url,
                model=self.model,
            )
            if condensed:
                return RewrittenQuery(
                    search_query=condensed[:CONVERSATION_RETRIEVAL_QUERY_MAX_CHARS],
                    references_prior_turn=True,
                )
        except Exception as exc:
            log.warning("ollama_rewrite_query_fallback", error=str(exc))
        return RewrittenQuery(search_query=latest, references_prior_turn=False)

    async def route(self, message: str) -> RouteDecision:
        # Heuristic carries the greeting fast-path; only escalate to the LLM when
        # the cheap rule is unsure between direct vs RAG (i.e. it labelled it direct
        # by length but the model may know better).
        baseline = heuristic_route(message)
        if baseline.route != ROUTE_DIRECT:
            return baseline
        try:
            return await ollama_provider.decide_route(
                message, base_url=self.base_url, model=self.model
            )
        except Exception as exc:
            log.warning("ollama_route_fallback", error=str(exc))
            return baseline

    async def grade(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        threshold: float,
        conversation: str | None = None,
    ) -> RetrievalGrade:
        try:
            return await ollama_provider.grade_retrieval(
                query=query,
                chunks=chunks,
                threshold=threshold,
                base_url=self.base_url,
                model=self.model,
                conversation=conversation,
            )
        except Exception as exc:
            log.warning("ollama_grade_fallback", error=str(exc))
            return await self._fallback.grade(
                query=query,
                chunks=chunks,
                threshold=threshold,
                conversation=conversation,
            )

    async def generate(
        self,
        *,
        query: str,
        contexts: list[str],
        route: RouteKind,
        chat_history: list[dict[str, str]] | None = None,
        retrieval_query: str | None = None,
    ) -> str:
        if route == ROUTE_DIRECT or not contexts:
            return heuristic_generate(
                query=query,
                contexts=contexts,
                route=route,
                chat_history=chat_history,
            )
        try:
            return await ollama_provider.generate_from_context(
                query=query,
                contexts=contexts,
                route=route,
                base_url=self.base_url,
                model=self.model,
                chat_history=chat_history,
                retrieval_query=retrieval_query,
            )
        except Exception as exc:
            log.warning("ollama_generate_fallback", error=str(exc))
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
        try:
            return await ollama_provider.validate_answer(
                answer=answer,
                contexts=contexts,
                base_url=self.base_url,
                model=self.model,
            )
        except Exception as exc:
            log.warning("ollama_validate_fallback", error=str(exc))
            return heuristic_validate(answer, contexts)

    async def contextualize(
        self,
        *,
        document_title: str,
        section: str,
        chunk: str,
        full_document: str,
    ) -> str:
        static = f"Document: {document_title} > Section: {section}"
        try:
            blurb = (
                await ollama_provider.contextualize_chunk(
                    document_title=document_title,
                    section=section,
                    chunk=chunk,
                    full_document=full_document,
                    base_url=self.base_url,
                    model=self.model,
                )
                or ""
            ).strip()
            return blurb if blurb else static
        except Exception as exc:  # degrade to structural prefix
            log.warning("ollama_contextualize_fallback", error=str(exc))
            return static
