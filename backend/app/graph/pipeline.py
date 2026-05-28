"""Run the LangGraph agent with DB + Qdrant dependencies."""

from __future__ import annotations

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.builder import invoke_agent_graph
from app.graph.state import GraphState
from app.retrieval.qdrant_store import QdrantStore

log = structlog.get_logger(__name__)


async def run_agent_pipeline(
    query: str,
    db: AsyncSession,
    qdrant: QdrantStore,
    *,
    checkpointer: object | None = None,
    compiled_graph: object | None = None,
    thread_id: str | None = None,
    retrieval_query: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> GraphState:
    state = await invoke_agent_graph(
        query,
        db=db,
        qdrant=qdrant,
        checkpointer=checkpointer,
        compiled_graph=compiled_graph,
        thread_id=thread_id,
        retrieval_query=retrieval_query,
        chat_history=chat_history,
    )
    log.info(
        "pipeline_complete",
        route=state.get("route"),
        abstained=state.get("abstained"),
        nodes=state.get("nodes_visited"),
    )
    return state
