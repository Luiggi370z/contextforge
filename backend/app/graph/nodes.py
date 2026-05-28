from __future__ import annotations

import structlog

from app.core.config import get_settings
from app.core.constants import (
    ABSTAIN_MESSAGE,
    CITATION_SNIPPET_MAX_CHARS,
    GRAPH_NODE_GENERATE,
    GRAPH_NODE_GRADE,
    GRAPH_NODE_RETRIEVE,
    GRAPH_NODE_ROUTE,
    GRAPH_NODE_VALIDATE,
    ROUTE_DIRECT,
    ROUTE_SINGLE_HOP_RAG,
)
from app.graph.chunks import chunks_from_state
from app.graph.state import GraphState
from app.llm.grading import grade_retrieval, select_chunks_for_generation
from app.llm.structured import (
    decide_route,
    generate_from_context,
    validate_answer,
)
from app.retrieval.dedupe import dedupe_citations
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
            "chunk_id": str(chunk.chunk_id),
            "document_id": str(chunk.document_id),
            "content": chunk.content,
            "score": chunk.score,
            "relevance_score": chunk.relevance_score,
        }
        for chunk in chunks
    ]
    return {
        "documents": documents,
        "retrieval_scores": [chunk.score for chunk in chunks],
        "nodes_visited": visited,
    }


async def grade_node(state: GraphState, *, chunks: list[RetrievedChunk]) -> dict:
    visited = list(state.get("nodes_visited", []))
    visited.append(GRAPH_NODE_GRADE)
    grade = await grade_retrieval(
        chunks,
        state.get("query", ""),
        retrieval_query=state.get("retrieval_query"),
        chat_history=state.get("chat_history"),
    )
    return {
        "abstained": grade.should_abstain,
        "nodes_visited": visited,
    }


async def generate_node(state: GraphState) -> dict:
    visited = list(state.get("nodes_visited", []))
    visited.append(GRAPH_NODE_GENERATE)
    route = state.get("route", ROUTE_SINGLE_HOP_RAG)
    docs = state.get("documents", [])
    query = state.get("query", "")
    chunks = chunks_from_state(state)
    selected_chunks = select_chunks_for_generation(chunks)
    contexts = [chunk.content for chunk in selected_chunks]
    chat_history = state.get("chat_history") or []
    if state.get("abstained"):
        answer = ABSTAIN_MESSAGE
    else:
        answer = await generate_from_context(
            query,
            contexts,
            route,  # type: ignore[arg-type]
            chat_history=chat_history,
        )
    citations: list[dict] = []
    if not state.get("abstained"):
        citations = dedupe_citations(
            [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "snippet": chunk.content[:CITATION_SNIPPET_MAX_CHARS],
                    "score": chunk.score,
                }
                for chunk in selected_chunks
            ]
        )
    return {"answer": answer, "citations": citations, "nodes_visited": visited}


async def validate_node(state: GraphState) -> dict:
    visited = list(state.get("nodes_visited", []))
    visited.append(GRAPH_NODE_VALIDATE)
    if state.get("route") == ROUTE_DIRECT:
        return {"nodes_visited": visited}
    docs = state.get("documents", [])
    contexts = [document["content"] for document in docs]
    validation = validate_answer(state.get("answer", ""), contexts)
    if not validation.grounded and not state.get("abstained"):
        settings = get_settings()
        if settings.llm_provider == "ollama":
            log.warning("validate_not_grounded_ollama", issues=validation.issues)
            return {"nodes_visited": visited}
        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "citations": [],
            "nodes_visited": visited,
        }
    return {"nodes_visited": visited}
