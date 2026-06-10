"""Structured outputs every LLM provider returns through the LLMProvider Protocol."""

from typing import Literal

from pydantic import BaseModel, Field

RouteKind = Literal["direct", "single_hop_rag", "multi_hop"]


class RouteDecision(BaseModel):
    """Routing decision from `LLMProvider.route`."""

    route: RouteKind
    confidence: float = Field(ge=0, le=1, default=0.8)
    reasoning: str = ""


class RetrievalGrade(BaseModel):
    """Output of `LLMProvider.grade` — should we proceed to generation?

    ``score`` is the rerank score (or earlier-stage score via
    ``RetrievedChunk.ranking_score()``). Cross-encoder logits can be negative,
    so we don't bound it on the low side; abstention is driven by the
    backend-specific threshold in :mod:`app.llm.grading`, not by sign.
    """

    relevant: bool
    score: float
    should_abstain: bool


class AnswerValidation(BaseModel):
    """Output of `LLMProvider.validate` — is the answer entailed by the context?"""

    grounded: bool
    issues: list[str] = Field(default_factory=list)
