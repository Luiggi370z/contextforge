"""LangGraph workflow: route → retrieve → grade → generate → validate."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.core.constants import ROUTE_DIRECT
from app.graph import nodes
from app.graph.chunks import chunks_from_state
from app.graph.state import GraphState
from app.retrieval.hybrid import hybrid_retrieve
from app.retrieval.models import RetrievedChunk

log = structlog.get_logger(__name__)


def _cfg(config: RunnableConfig) -> dict[str, Any]:
    return config.get("configurable") or {}


async def _route(state: GraphState, config: RunnableConfig) -> dict:
    return await nodes.route_node(state)


def _after_route(state: GraphState) -> Literal["retrieve", "generate"]:
    if state.get("route") == ROUTE_DIRECT:
        return "generate"
    return "retrieve"


async def _retrieve(state: GraphState, config: RunnableConfig) -> dict:
    cfg = _cfg(config)
    db = cfg["db"]
    qdrant = cfg["qdrant"]
    query = state.get("query", "")
    retrieval_query = state.get("retrieval_query") or query
    route = state.get("route", "single_hop_rag")

    # TODO(retrieval-backend): use hybrid_retrieve_configured(db, query, qdrant=qdrant)
    chunks: list[RetrievedChunk] = await hybrid_retrieve(db, qdrant, retrieval_query)
    if route == "multi_hop" and len(chunks) > 2:
        initial_count = len(chunks)
        extra = await hybrid_retrieve(db, qdrant, f"{retrieval_query} details")
        seen = {str(chunk.chunk_id) for chunk in chunks}
        for chunk in extra:
            if str(chunk.chunk_id) not in seen:
                chunks.append(chunk)
                seen.add(str(chunk.chunk_id))
        chunks = chunks[: max(3, initial_count // 2 + 3)]

    out = await nodes.retrieve_node(state, chunks=chunks)
    out["_chunks"] = chunks
    return out


async def _grade(state: GraphState, config: RunnableConfig) -> dict:
    chunks = chunks_from_state(state)
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


def _graph_run_config(
    *,
    db: Any,
    qdrant: Any,
    thread_id: str | None = None,
) -> RunnableConfig:
    configurable: dict[str, Any] = {"db": db, "qdrant": qdrant}
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    return {"configurable": configurable}


def _initial_graph_state(
    query: str,
    *,
    retrieval_query: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    trace_id: str | None = None,
) -> GraphState:
    return {
        "query": query,
        "retrieval_query": retrieval_query or query,
        "chat_history": chat_history or [],
        "nodes_visited": [],
        "documents": [],
        "citations": [],
        "abstained": False,
        "retrieval_scores": [],
        "trace_id": trace_id or "",
    }


def _finalize_graph_state(state: GraphState) -> GraphState:
    if isinstance(state, dict):
        state.pop("_chunks", None)
        state.pop("_selected_chunks", None)
    return state


async def invoke_agent_graph(
    query: str,
    *,
    db: Any,
    qdrant: Any,
    checkpointer: Any | None = None,
    compiled_graph: Any | None = None,
    thread_id: str | None = None,
    retrieval_query: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    trace_id: str | None = None,
) -> GraphState:
    """Run the agent graph; ``thread_id`` enables Postgres checkpoint resume."""
    compiled = compiled_graph or build_agent_graph(checkpointer=checkpointer)
    config = _graph_run_config(db=db, qdrant=qdrant, thread_id=thread_id)
    result = await compiled.ainvoke(
        _initial_graph_state(
            query,
            retrieval_query=retrieval_query,
            chat_history=chat_history,
            trace_id=trace_id or thread_id,
        ),
        config=config,
    )
    return _finalize_graph_state(result)  # type: ignore[return-value]


async def stream_invoke_agent_graph(
    query: str,
    *,
    db: Any,
    qdrant: Any,
    checkpointer: Any | None = None,
    compiled_graph: Any | None = None,
    thread_id: str | None = None,
    retrieval_query: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    trace_id: str | None = None,
) -> AsyncIterator[str | GraphState]:
    """Yield each completed node name, then the final graph state."""
    compiled = compiled_graph or build_agent_graph(checkpointer=checkpointer)
    config = _graph_run_config(db=db, qdrant=qdrant, thread_id=thread_id)
    state: GraphState = _initial_graph_state(
        query,
        retrieval_query=retrieval_query,
        chat_history=chat_history,
        trace_id=trace_id or thread_id,
    )
    async for update in compiled.astream(state, config=config, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node_name, node_update in update.items():
            if isinstance(node_update, dict):
                state = {**state, **node_update}  # type: ignore[typeddict-item]
            yield node_name
    yield _finalize_graph_state(state)
