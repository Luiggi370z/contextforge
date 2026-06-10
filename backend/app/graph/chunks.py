"""Reconstruct ``RetrievedChunk`` objects from serialized graph state."""

from __future__ import annotations

from uuid import UUID

from app.graph.state import GraphState
from app.retrieval.models import RetrievedChunk


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def chunks_from_state(state: GraphState) -> list[RetrievedChunk]:
    """Prefer the in-memory ``_chunks`` payload; rebuild from documents otherwise."""
    stored = list(state.get("_chunks") or [])
    if stored:
        return stored
    rebuilt: list[RetrievedChunk] = []
    for document in state.get("documents", []):
        chunk_id = document.get("chunk_id")
        document_id = document.get("document_id")
        content = document.get("content")
        if chunk_id is None or document_id is None or content is None:
            continue
        rebuilt.append(
            RetrievedChunk(
                chunk_id=UUID(str(chunk_id)),
                document_id=UUID(str(document_id)),
                content=str(content),
                score=float(document.get("score", 0.0)),
                metadata=dict(document.get("metadata") or {}),
                dense_score=_opt_float(document.get("dense_score")),
                sparse_score=_opt_float(document.get("sparse_score")),
                rrf_score=_opt_float(document.get("rrf_score")),
                rerank_score=_opt_float(document.get("rerank_score")),
            )
        )
    return rebuilt
