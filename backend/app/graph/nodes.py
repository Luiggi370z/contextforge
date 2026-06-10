"""Graph node business logic.

State invariant after Phase 2:

- ``documents`` carries every chunk we retrieved, with every stage's score.
- ``_chunks`` (internal) is the same list as ``RetrievedChunk`` objects.
- ``_selected_chunks`` (internal) is the *subset* used for generation and
  validation; citations are emitted from this list. ``selected_indices`` is
  the API-visible projection so the frontend can highlight which retrieved
  chunks the answer actually drew on.
"""

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


def _bind(state: GraphState):
    """Return a logger pre-bound with trace_id so every node log is traceable."""
    return log.bind(trace_id=state.get("trace_id") or "")


def _chunk_to_document(chunk: RetrievedChunk) -> dict:
    return {
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(chunk.document_id),
        "content": chunk.content,
        "score": chunk.ranking_score(),
        "metadata": chunk.metadata or {},
        "dense_score": chunk.dense_score,
        "sparse_score": chunk.sparse_score,
        "rrf_score": chunk.rrf_score,
        "rerank_score": chunk.rerank_score,
    }


async def route_node(state: GraphState) -> dict:
    query = state.get("query", "")
    decision = await decide_route(query)
    visited = list(state.get("nodes_visited", []))
    visited.append(GRAPH_NODE_ROUTE)
    _bind(state).info(
        "graph_route", route=decision.route, confidence=decision.confidence
    )
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
    documents = [_chunk_to_document(chunk) for chunk in chunks]
    _bind(state).info(
        "graph_retrieve",
        candidates=len(chunks),
        top_score=chunks[0].ranking_score() if chunks else 0.0,
    )
    return {
        "documents": documents,
        "retrieval_scores": [chunk.ranking_score() for chunk in chunks],
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
    _bind(state).info(
        "graph_grade",
        relevant=grade.relevant,
        score=grade.score,
        should_abstain=grade.should_abstain,
    )
    return {
        "abstained": grade.should_abstain,
        "nodes_visited": visited,
    }


async def generate_node(state: GraphState) -> dict:
    """Pick the chunks that actually inform the answer, then generate + cite from them.

    Generation, citations, and validation all reference the same
    ``selected_chunks`` list so we cannot ship a citation that the model never
    saw, nor validate against context the model never received.
    """
    visited = list(state.get("nodes_visited", []))
    visited.append(GRAPH_NODE_GENERATE)
    route = state.get("route", ROUTE_SINGLE_HOP_RAG)
    query = state.get("query", "")
    retrieval_query = state.get("retrieval_query") or query
    chunks = chunks_from_state(state)
    selected_chunks = select_chunks_for_generation(chunks)
    contexts = [chunk.content for chunk in selected_chunks]
    chat_history = state.get("chat_history") or []
    if state.get("abstained"):
        answer = ABSTAIN_MESSAGE
        selected_chunks = []
        citations: list[dict] = []
    else:
        answer = await generate_from_context(
            query,
            contexts,
            route,  # type: ignore[arg-type]
            chat_history=chat_history,
            retrieval_query=retrieval_query,
        )
        citations = dedupe_citations(
            [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "snippet": chunk.content[:CITATION_SNIPPET_MAX_CHARS],
                    "score": chunk.ranking_score(),
                }
                for chunk in selected_chunks
            ]
        )
    _bind(state).info(
        "graph_generate",
        selected=len(selected_chunks),
        abstained=bool(state.get("abstained")),
    )
    return {
        "answer": answer,
        "citations": citations,
        "_selected_chunks": selected_chunks,
        "nodes_visited": visited,
    }


async def validate_node(state: GraphState) -> dict:
    """Re-check grounding against the *same* contexts generation used."""
    visited = list(state.get("nodes_visited", []))
    visited.append(GRAPH_NODE_VALIDATE)
    if state.get("route") == ROUTE_DIRECT or state.get("abstained"):
        return {"nodes_visited": visited}
    selected = state.get("_selected_chunks") or []
    contexts = [
        chunk.content if hasattr(chunk, "content") else chunk["content"]
        for chunk in selected
    ]
    validation = await validate_answer(state.get("answer", ""), contexts)
    if not validation.grounded:
        _bind(state).warning("validate_not_grounded", issues=validation.issues)
        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "citations": [],
            "nodes_visited": visited,
        }
    return {"nodes_visited": visited}
