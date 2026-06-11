"""OpenAI-backed provider: gpt-4o-mini by default, Instructor for structured outputs."""

from __future__ import annotations

import structlog

from app.core.constants import (
    CONVERSATION_RETRIEVAL_QUERY_MAX_CHARS,
    ROUTE_DIRECT,
)
from app.graph.conversation import ChatTurn, format_chat_history
from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision, RouteKind
from app.llm.prompts.openai import (
    SYSTEM_PROMPT_GENERATE,
    SYSTEM_PROMPT_GRADE,
    SYSTEM_PROMPT_RETRIEVAL_QUERY,
    SYSTEM_PROMPT_ROUTE,
    SYSTEM_PROMPT_VALIDATE,
    build_generate_user_prompt,
    build_grade_user_prompt,
    build_retrieval_query_user_prompt,
    build_validate_user_prompt,
)
from app.llm.providers.base import RewrittenQuery
from app.llm.providers.heuristic import (
    HeuristicProvider,
    heuristic_generate,
    heuristic_route,
    heuristic_validate,
)
from app.retrieval.models import RetrievedChunk

log = structlog.get_logger(__name__)


def _structured_client():
    """Build an Instructor-wrapped Async OpenAI client (lazy import keeps tests light)."""
    import instructor
    from openai import AsyncOpenAI

    from app.core.config import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("openai_api_key_missing")
    return instructor.from_openai(AsyncOpenAI(api_key=settings.openai_api_key))


def _chat_client():
    """Plain AsyncOpenAI client for free-text generation."""
    from openai import AsyncOpenAI

    from app.core.config import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("openai_api_key_missing")
    return AsyncOpenAI(api_key=settings.openai_api_key)


class OpenAIProvider:
    """LLMProvider backed by OpenAI (Instructor for structured outputs)."""

    name = "openai"

    def __init__(self, *, model: str) -> None:
        self.model = model
        self._fallback = HeuristicProvider()

    async def rewrite_query(
        self, history: list[ChatTurn], latest: str
    ) -> RewrittenQuery:
        latest = latest.strip()
        if not history:
            return RewrittenQuery(search_query=latest, references_prior_turn=False)
        try:
            client = _structured_client()
            conversation = format_chat_history(history)
            result: RewrittenQuery = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_RETRIEVAL_QUERY},
                    {
                        "role": "user",
                        "content": build_retrieval_query_user_prompt(
                            conversation=conversation,
                            latest_message=latest,
                        ),
                    },
                ],
                response_model=RewrittenQuery,
            )
            result.search_query = result.search_query.strip()[
                :CONVERSATION_RETRIEVAL_QUERY_MAX_CHARS
            ]
            return result
        except Exception as exc:
            log.warning("openai_rewrite_query_fallback", error=str(exc))
        return RewrittenQuery(search_query=latest, references_prior_turn=False)

    async def route(self, message: str) -> RouteDecision:
        baseline = heuristic_route(message)
        if baseline.route != ROUTE_DIRECT:
            return baseline
        try:
            client = _structured_client()
            return await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ROUTE},
                    {"role": "user", "content": message},
                ],
                response_model=RouteDecision,
            )
        except Exception as exc:
            log.warning("openai_route_fallback", error=str(exc))
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
            client = _structured_client()
            return await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_GRADE},
                    {
                        "role": "user",
                        "content": build_grade_user_prompt(
                            query=query,
                            threshold=threshold,
                            contexts=[chunk.content for chunk in chunks],
                            conversation=conversation,
                        ),
                    },
                ],
                response_model=RetrievalGrade,
            )
        except Exception as exc:
            log.warning("openai_grade_fallback", error=str(exc))
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
            client = _chat_client()
            conversation: str | None = None
            if chat_history and len(chat_history) > 1:
                prior = [
                    ChatTurn(role=turn["role"], content=turn["content"])
                    for turn in chat_history[:-1]
                ]
                conversation = format_chat_history(prior)
            user_prompt = build_generate_user_prompt(
                route=route,
                query=query,
                contexts=contexts,
                conversation=conversation,
                retrieval_query=retrieval_query,
            )
            completion = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_GENERATE},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:
            log.warning("openai_generate_fallback", error=str(exc))
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
            client = _structured_client()
            return await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_VALIDATE},
                    {
                        "role": "user",
                        "content": build_validate_user_prompt(
                            answer=answer, contexts=contexts
                        ),
                    },
                ],
                response_model=AnswerValidation,
            )
        except Exception as exc:
            log.warning("openai_validate_fallback", error=str(exc))
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
        system_prompt = (
            "You situate a document chunk within its source document for search "
            "retrieval. Answer with ONLY a short 1-2 sentence context. No preamble, "
            "no quotes, no labels."
        )
        user_prompt = (
            "Write a 1-2 sentence context that situates the following chunk within "
            "the document, to improve search retrieval. Answer with ONLY the "
            "context.\n\n"
            f"Document title: {document_title}\n"
            f"Section: {section}\n\n"
            f"<document>\n{full_document[:8000]}\n</document>\n\n"
            f"<chunk>\n{chunk}\n</chunk>"
        )
        try:
            client = _chat_client()
            completion = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            blurb = (completion.choices[0].message.content or "").strip()
            return blurb if blurb else static
        except Exception as exc:  # degrade to structural prefix
            log.warning("openai_contextualize_fallback", error=str(exc))
            return static
