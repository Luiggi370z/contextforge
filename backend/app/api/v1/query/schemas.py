import uuid
from typing import Literal

from pydantic import BaseModel, Field

RouteKind = Literal["direct", "single_hop_rag", "multi_hop", "unknown"]


class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: uuid.UUID | None = None
    # TODO(retrieval-backend-ui): optional retrieval_backend: Literal["qdrant", "postgres"] | None
    # so the React settings toggle can override server default per query without restart.


class Citation(BaseModel):
    document_id: uuid.UUID | None = None
    chunk_id: uuid.UUID | None = None
    snippet: str
    score: float | None = None


class QueryMetadata(BaseModel):
    route: RouteKind = "unknown"
    abstained: bool = False
    nodes_visited: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    metadata: QueryMetadata = Field(default_factory=QueryMetadata)
