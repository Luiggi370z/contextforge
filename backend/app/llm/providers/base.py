"""LLM provider Protocol and shared output models."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from app.graph.conversation import ChatTurn
from app.llm.models import AnswerValidation, RetrievalGrade, RouteDecision, RouteKind
from app.retrieval.models import RetrievedChunk


class RewrittenQuery(BaseModel):
    """Standalone search query derived from a multi-turn conversation."""

    search_query: str = Field(min_length=1)
    references_prior_turn: bool = False


class LLMProvider(Protocol):
    """Single abstraction every graph node calls; dispatch by ``LLM_PROVIDER``."""

    name: str

    async def rewrite_query(
        self,
        history: list[ChatTurn],
        latest: str,
    ) -> RewrittenQuery: ...

    async def route(self, message: str) -> RouteDecision: ...

    async def grade(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        threshold: float,
        conversation: str | None = None,
    ) -> RetrievalGrade: ...

    async def generate(
        self,
        *,
        query: str,
        contexts: list[str],
        route: RouteKind,
        chat_history: list[dict[str, str]] | None = None,
        retrieval_query: str | None = None,
    ) -> str: ...

    async def validate(
        self,
        *,
        answer: str,
        contexts: list[str],
    ) -> AnswerValidation: ...
