"""Helpers to rebuild retrieved chunks from graph state."""

from __future__ import annotations

from uuid import UUID

from app.graph.state import GraphState
from app.retrieval.models import RetrievedChunk


def chunks_from_state(state: GraphState) -> list[RetrievedChunk]:
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
        score = float(document.get("score", 0.0))
        relevance = document.get("relevance_score")
        rebuilt.append(
            RetrievedChunk(
                chunk_id=UUID(str(chunk_id)),
                document_id=UUID(str(document_id)),
                content=str(content),
                score=score,
                relevance_score=float(relevance) if relevance is not None else None,
            )
        )
    return rebuilt
