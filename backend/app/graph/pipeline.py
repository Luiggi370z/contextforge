"""Agent pipeline orchestrating route → retrieve → grade → generate → validate."""

from __future__ import annotations

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph import nodes
from app.graph.state import GraphState
from app.retrieval.hybrid import RetrievedChunk, hybrid_retrieve
from app.retrieval.qdrant_store import QdrantStore

log = structlog.get_logger(__name__)


async def run_agent_pipeline(
    query: str,
    db: AsyncSession,
    qdrant: QdrantStore,
) -> GraphState:
    state: GraphState = {
        "query": query,
        "nodes_visited": [],
        "documents": [],
        "citations": [],
        "abstained": False,
        "retrieval_scores": [],
    }

    route_update = await nodes.route_node(state)
    state.update(route_update)
    route = state.get("route", "single_hop_rag")

    chunks: list[RetrievedChunk] = []
    if route in ("single_hop_rag", "multi_hop"):
        chunks = await hybrid_retrieve(db, qdrant, query)
        if route == "multi_hop" and len(chunks) > 2:
            half = max(1, len(chunks) // 2)
            extra = await hybrid_retrieve(db, qdrant, f"{query} details")
            seen = {str(c.chunk_id) for c in chunks}
            for c in extra:
                if str(c.chunk_id) not in seen:
                    chunks.append(c)
                    seen.add(str(c.chunk_id))
            chunks = chunks[: half + 3]
        state.update(await nodes.retrieve_node(state, chunks=chunks))
        state.update(await nodes.grade_node(state, chunks=chunks))

    state.update(await nodes.generate_node(state))
    state.update(await nodes.validate_node(state))

    log.info(
        "pipeline_complete",
        route=route,
        abstained=state.get("abstained"),
        nodes=state.get("nodes_visited"),
    )
    return state
