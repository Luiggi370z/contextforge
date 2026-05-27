from __future__ import annotations

import structlog

from app.core.constants import (
    ABSTAIN_MESSAGE,
    CITATION_SNIPPET_MAX_CHARS,
    GRAPH_NODE_GENERATE,
    GRAPH_NODE_GRADE,
    GRAPH_NODE_RETRIEVE,
    GRAPH_NODE_ROUTE,
    GRAPH_NODE_VALIDATE,
    ROUTE_SINGLE_HOP_RAG,
)
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
    visited.append(GRAPH_NODE_ROUTE)
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
    visited.append(GRAPH_NODE_RETRIEVE)
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
    visited.append(GRAPH_NODE_GRADE)
    grade = grade_retrieval(chunks, state.get("query", ""))
    return {
        "abstained": grade.should_abstain,
        "nodes_visited": visited,
    }


async def generate_node(state: GraphState) -> dict:
    visited = list(state.get("nodes_visited", []))
    visited.append(GRAPH_NODE_GENERATE)
    route = state.get("route", ROUTE_SINGLE_HOP_RAG)
    docs = state.get("documents", [])
    contexts = [d["content"] for d in docs]
    if state.get("abstained"):
        answer = ABSTAIN_MESSAGE
    else:
        answer = generate_from_context(state.get("query", ""), contexts, route)  # type: ignore[arg-type]
    citations = [
        {
            "chunk_id": d.get("chunk_id"),
            "document_id": d.get("document_id"),
            "snippet": d.get("content", "")[:CITATION_SNIPPET_MAX_CHARS],
            "score": d.get("score"),
        }
        for d in docs
    ]
    return {"answer": answer, "citations": citations, "nodes_visited": visited}


async def validate_node(state: GraphState) -> dict:
    visited = list(state.get("nodes_visited", []))
    visited.append(GRAPH_NODE_VALIDATE)
    docs = state.get("documents", [])
    contexts = [d["content"] for d in docs]
    validation = validate_answer(state.get("answer", ""), contexts)
    if not validation.grounded and not state.get("abstained"):
        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "nodes_visited": visited,
        }
    return {"nodes_visited": visited}
