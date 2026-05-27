from typing import Literal

from pydantic import BaseModel, Field

RouteKind = Literal["direct", "single_hop_rag", "multi_hop"]


class RouteDecision(BaseModel):
    route: RouteKind
    confidence: float = Field(ge=0, le=1, default=0.8)
    reasoning: str = ""


class RetrievalGrade(BaseModel):
    relevant: bool
    score: float = Field(ge=0, le=1)
    should_abstain: bool


class AnswerValidation(BaseModel):
    grounded: bool
    issues: list[str] = Field(default_factory=list)


class SubQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=4)
