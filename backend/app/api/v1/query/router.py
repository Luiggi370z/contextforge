"""HTTP routes for RAG queries (sync and SSE)."""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.metrics.service import increment_queries
from app.api.v1.query.dependencies import (
    get_compiled_agent_graph,
    get_langgraph_checkpointer,
    get_query_service,
)
from app.api.v1.query.schemas import QueryRequest, QueryResponse
from app.api.v1.query.service import QueryService
from app.db.session import get_db

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("", response_model=QueryResponse)
async def query_sync(
    body: QueryRequest,
    session: AsyncSession = Depends(get_db),
    query_service: QueryService = Depends(get_query_service),
    checkpointer: object | None = Depends(get_langgraph_checkpointer),
    compiled_graph: object | None = Depends(get_compiled_agent_graph),
) -> QueryResponse:
    """Run a query and return the full JSON response."""
    increment_queries()
    return await query_service.execute(
        body,
        session,
        checkpointer=checkpointer,
        compiled_graph=compiled_graph,
    )


@router.post("/stream")
async def query_stream(
    body: QueryRequest,
    session: AsyncSession = Depends(get_db),
    query_service: QueryService = Depends(get_query_service),
    checkpointer: object | None = Depends(get_langgraph_checkpointer),
    compiled_graph: object | None = Depends(get_compiled_agent_graph),
) -> StreamingResponse:
    """Stream graph status, answer tokens, and final result with citations as SSE."""
    increment_queries()
    return StreamingResponse(
        query_service.stream_execute(
            body,
            session,
            checkpointer=checkpointer,
            compiled_graph=compiled_graph,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
