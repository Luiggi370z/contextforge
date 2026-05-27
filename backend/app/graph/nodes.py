from __future__ import annotations

import structlog

from app.graph.state import GraphState
from app.llm.structured import (
    decide_route,
    generate_from_context,
    grade_retrieval,
    validate_answer,
)
from app.retrieval.models import RetrievedChunk

log = structlog.get_logger(__name__)


async def route_node(state: GraphState) -> dict:
    query = state.get("query", "")
    decision = await decide_route(query)
    visited = list(state.get("nodes_visited", []))
    visited.append("route")
    log.info("graph_route", route=decision.route, confidence=decision.confidence)
    return {
        "route": decision.route,
        "nodes_visited": visited,
    }


async def retrieve_node(
    state: GraphState,
    *,
    chunks: list[RetrievedChunk],
) -> dict:
    visited = list(state.get("nodes_visited", []))
    visited.append("retrieve")
    documents = [
        {
            "chunk_id": str(c.chunk_id),
            "document_id": str(c.document_id),
            "content": c.content,
            "score": c.score,
        }
        for c in chunks
    ]
    return {
        "documents": documents,
        "retrieval_scores": [c.score for c in chunks],
        "nodes_visited": visited,
    }


async def grade_node(state: GraphState, *, chunks: list[RetrievedChunk]) -> dict:
    visited = list(state.get("nodes_visited", []))
    visited.append("grade_context")
    grade = grade_retrieval(chunks, state.get("query", ""))
    return {
        "abstained": grade.should_abstain,
        "nodes_visited": visited,
    }


async def generate_node(state: GraphState) -> dict:
    visited = list(state.get("nodes_visited", []))
    visited.append("generate")
    route = state.get("route", "single_hop_rag")
    docs = state.get("documents", [])
    contexts = [d["content"] for d in docs]
    if state.get("abstained"):
        answer = "I don't have enough information in the indexed documents to answer that."
    else:
        answer = generate_from_context(state.get("query", ""), contexts, route)  # type: ignore[arg-type]
    citations = [
        {
            "chunk_id": d.get("chunk_id"),
            "document_id": d.get("document_id"),
            "snippet": d.get("content", "")[:240],
            "score": d.get("score"),
        }
        for d in docs
    ]
    return {"answer": answer, "citations": citations, "nodes_visited": visited}


async def validate_node(state: GraphState) -> dict:
    visited = list(state.get("nodes_visited", []))
    visited.append("validate_answer")
    docs = state.get("documents", [])
    contexts = [d["content"] for d in docs]
    validation = validate_answer(state.get("answer", ""), contexts)
    if not validation.grounded and not state.get("abstained"):
        return {
            "answer": "I don't have enough information in the indexed documents to answer that.",
            "abstained": True,
            "nodes_visited": visited,
        }
    return {"nodes_visited": visited}
