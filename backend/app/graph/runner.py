from __future__ import annotations

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message, Thread
from app.graph.pipeline import run_agent_pipeline
from app.retrieval.qdrant_store import get_qdrant_store
from app.api.v1.query.schemas import Citation, QueryMetadata, QueryRequest, QueryResponse

log = structlog.get_logger(__name__)


async def run_query(body: QueryRequest, db: AsyncSession) -> QueryResponse:
    thread_id = body.thread_id
    if thread_id is None:
        thread = Thread(title=body.message[:80])
        db.add(thread)
        await db.commit()
        await db.refresh(thread)
        thread_id = thread.id
    else:
        thread = await db.get(Thread, thread_id)
        if thread is None:
            thread = Thread(id=thread_id, title=body.message[:80])
            db.add(thread)
            await db.commit()

    db.add(Message(thread_id=thread_id, role="user", content=body.message))
    await db.commit()

    qdrant = get_qdrant_store()
    state = await run_agent_pipeline(body.message, db, qdrant)

    citations = [
        Citation(
            document_id=c.get("document_id"),
            chunk_id=c.get("chunk_id"),
            snippet=c.get("snippet", ""),
            score=c.get("score"),
        )
        for c in state.get("citations", [])
    ]
    metadata = QueryMetadata(
        route=state.get("route", "unknown"),  # type: ignore[arg-type]
        abstained=bool(state.get("abstained")),
        nodes_visited=list(state.get("nodes_visited", [])),
        retrieval_scores=list(state.get("retrieval_scores", [])),
    )
    response = QueryResponse(
        answer=state.get("answer", ""),
        citations=citations,
        metadata=metadata,
    )

    db.add(
        Message(
            thread_id=thread_id,
            role="assistant",
            content=response.answer,
            metadata_=response.metadata.model_dump(),
        )
    )
    await db.commit()
    log.info("query_complete", thread_id=str(thread_id), route=metadata.route)
    return response
