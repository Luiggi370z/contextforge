from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.query.schemas import Citation, QueryMetadata, QueryRequest, QueryResponse
from app.core.constants import THREAD_TITLE_MAX_CHARS
from app.db.models import Message, Thread
from app.graph.pipeline import run_agent_pipeline
from app.retrieval.qdrant_store import get_qdrant_store

log = structlog.get_logger(__name__)


async def _ensure_thread(session: AsyncSession, body: QueryRequest) -> UUID:
    """Create or load a conversation thread for the query."""
    thread_id = body.thread_id
    if thread_id is None:
        thread = Thread(title=body.message[:THREAD_TITLE_MAX_CHARS])
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
        return thread.id

    thread = await session.get(Thread, thread_id)
    if thread is None:
        thread = Thread(id=thread_id, title=body.message[:THREAD_TITLE_MAX_CHARS])
        session.add(thread)
        await session.commit()
    return thread_id


async def run_query(
    body: QueryRequest,
    session: AsyncSession,
    *,
    checkpointer: Any | None = None,
    compiled_graph: Any | None = None,
) -> QueryResponse:
    """Persist messages, run LangGraph, and return a grounded response."""
    thread_id = await _ensure_thread(session, body)

    session.add(Message(thread_id=thread_id, role="user", content=body.message))
    await session.commit()

    # TODO(retrieval-backend): use get_vector_store() from app.retrieval.factory
    qdrant = get_qdrant_store()
    state = await run_agent_pipeline(
        body.message,
        session,
        qdrant,
        checkpointer=checkpointer,
        compiled_graph=compiled_graph,
        thread_id=str(thread_id),
    )

    citations = [
        Citation(
            document_id=citation.get("document_id"),
            chunk_id=citation.get("chunk_id"),
            snippet=citation.get("snippet", ""),
            score=citation.get("score"),
        )
        for citation in state.get("citations", [])
    ]
    metadata = QueryMetadata(
        route=state.get("route", "unknown"),  # type: ignore[arg-type]
        abstained=bool(state.get("abstained")),
        nodes_visited=list(state.get("nodes_visited", [])),
        retrieval_scores=list(state.get("retrieval_scores", [])),
        graph_checkpoint_enabled=checkpointer is not None,
    )
    response = QueryResponse(
        answer=state.get("answer", ""),
        thread_id=thread_id,
        citations=citations,
        metadata=metadata,
    )

    session.add(
        Message(
            thread_id=thread_id,
            role="assistant",
            content=response.answer,
            metadata_=response.metadata.model_dump(),
        )
    )
    await session.commit()
    log.info(
        "query_complete",
        thread_id=str(thread_id),
        route=metadata.route,
        checkpoint=metadata.graph_checkpoint_enabled,
    )
    return response
