import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.graph.runner import run_query
from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query_sync(body: QueryRequest, db: AsyncSession = Depends(get_db)) -> QueryResponse:
    return await run_query(body, db)


@router.post("/stream")
async def query_stream(body: QueryRequest, db: AsyncSession = Depends(get_db)) -> StreamingResponse:
    result = await run_query(body, db)

    async def event_generator():
        payload = {"type": "token", "content": result.answer}
        yield f"data: {json.dumps(payload)}\n\n"
        done = {"type": "done", "result": result.model_dump(mode="json")}
        yield f"data: {json.dumps(done)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
