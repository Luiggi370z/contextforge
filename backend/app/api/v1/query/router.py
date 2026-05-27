"""HTTP routes for RAG queries (sync and SSE)."""

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.metrics.service import increment_queries
from app.api.v1.query.dependencies import get_query_service
from app.api.v1.query.schemas import QueryRequest, QueryResponse
from app.api.v1.query.service import QueryService
from app.db.session import get_db

router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query_sync(
    body: QueryRequest,
    session: AsyncSession = Depends(get_db),
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """Run a query and return the full JSON response."""
    increment_queries()
    return await query_service.execute(body, session)


@router.post("/stream")
async def query_stream(
    body: QueryRequest,
    session: AsyncSession = Depends(get_db),
    query_service: QueryService = Depends(get_query_service),
) -> StreamingResponse:
    """Stream query tokens and final result as SSE."""
    increment_queries()
    result = await query_service.execute(body, session)

    async def event_generator():
        payload = {"type": "token", "content": result.answer}
        yield f"data: {json.dumps(payload)}\n\n"
        done = {"type": "done", "result": result.model_dump(mode="json")}
        yield f"data: {json.dumps(done)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
