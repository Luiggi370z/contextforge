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
) -> GraphState:
    state = await invoke_agent_graph(
        query,
        db=db,
        qdrant=qdrant,
        checkpointer=checkpointer,
    )
    log.info(
        "pipeline_complete",
        route=state.get("route"),
        abstained=state.get("abstained"),
        nodes=state.get("nodes_visited"),
    )
    return state
