"""LangGraph workflow: route → retrieve → grade → generate → validate."""

from __future__ import annotations

from typing import Any, Literal

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import GraphState
from app.retrieval.hybrid import RetrievedChunk, hybrid_retrieve

log = structlog.get_logger(__name__)


def _cfg(config: RunnableConfig) -> dict[str, Any]:
    return config.get("configurable") or {}


async def _route(state: GraphState, config: RunnableConfig) -> dict:
    return await nodes.route_node(state)


def _after_route(state: GraphState) -> Literal["retrieve", "generate"]:
    if state.get("route") == "direct":
        return "generate"
    return "retrieve"


async def _retrieve(state: GraphState, config: RunnableConfig) -> dict:
    cfg = _cfg(config)
    db = cfg["db"]
    qdrant = cfg["qdrant"]
    query = state.get("query", "")
    route = state.get("route", "single_hop_rag")

    chunks: list[RetrievedChunk] = await hybrid_retrieve(db, qdrant, query)
    if route == "multi_hop" and len(chunks) > 2:
        extra = await hybrid_retrieve(db, qdrant, f"{query} details")
        seen = {str(c.chunk_id) for c in chunks}
        for c in extra:
            if str(c.chunk_id) not in seen:
                chunks.append(c)
                seen.add(str(c.chunk_id))
        chunks = chunks[: max(3, len(chunks) // 2 + 3)]

    out = await nodes.retrieve_node(state, chunks=chunks)
    out["_chunks"] = chunks
    return out


async def _grade(state: GraphState, config: RunnableConfig) -> dict:
    chunks: list[RetrievedChunk] = list(state.get("_chunks") or [])
    return await nodes.grade_node(state, chunks=chunks)


async def _generate(state: GraphState, config: RunnableConfig) -> dict:
    return await nodes.generate_node(state)


async def _validate(state: GraphState, config: RunnableConfig) -> dict:
    return await nodes.validate_node(state)


def build_agent_graph(checkpointer: Any | None = None) -> Any:
    graph = StateGraph(GraphState)
    graph.add_node("route", _route)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("grade_context", _grade)
    graph.add_node("generate", _generate)
    graph.add_node("validate_answer", _validate)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        _after_route,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("retrieve", "grade_context")
    graph.add_edge("grade_context", "generate")
    graph.add_edge("generate", "validate_answer")
    graph.add_edge("validate_answer", END)

    return graph.compile(checkpointer=checkpointer)


async def invoke_agent_graph(
    query: str,
    *,
    db: Any,
    qdrant: Any,
    checkpointer: Any | None = None,
) -> GraphState:
    compiled = build_agent_graph(checkpointer=checkpointer)
    initial: GraphState = {
        "query": query,
        "nodes_visited": [],
        "documents": [],
        "citations": [],
        "abstained": False,
        "retrieval_scores": [],
    }
    config: RunnableConfig = {"configurable": {"db": db, "qdrant": qdrant}}
    result = await compiled.ainvoke(initial, config=config)
    if isinstance(result, dict):
        result.pop("_chunks", None)
    return result  # type: ignore[return-value]
