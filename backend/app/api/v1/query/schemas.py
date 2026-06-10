import uuid
from typing import Literal

from pydantic import Field

from app.schemas.base import BaseRequest, BaseResponse

RouteKind = Literal["direct", "single_hop_rag", "multi_hop", "unknown"]


class QueryRequest(BaseRequest):
    message: str = Field(..., min_length=1)
    thread_id: uuid.UUID | None = None
    # TODO(retrieval-backend-ui): optional retrieval_backend: Literal["qdrant", "postgres"] | None


class Citation(BaseResponse):
    document_id: uuid.UUID | None = None
    chunk_id: uuid.UUID | None = None
    snippet: str
    score: float | None = None


class QueryMetadata(BaseResponse):
    route: RouteKind = "unknown"
    abstained: bool = False
    nodes_visited: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    graph_checkpoint_enabled: bool = False


class QueryResponse(BaseResponse):
    answer: str
    thread_id: uuid.UUID | None = None
    citations: list[Citation] = Field(default_factory=list)
    metadata: QueryMetadata = Field(default_factory=QueryMetadata)
