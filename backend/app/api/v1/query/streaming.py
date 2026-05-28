"""SSE formatting and answer chunking for query streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from app.api.v1.query.schemas import QueryResponse
from app.core.constants import (
    SSE_EVENT_DONE,
    SSE_EVENT_ERROR,
    SSE_EVENT_STATUS,
    SSE_EVENT_TOKEN,
    SSE_FLUSH_COMMENT,
    SSE_STAGE_STARTED,
    SSE_STREAM_CHUNK_DELAY_SECONDS,
    SSE_STREAM_WORD_CHUNK_SIZE,
)
from app.schemas.errors import ErrorResponse


def format_sse_event(event_type: str, payload: dict[str, object]) -> str:
    """Serialize one SSE ``data:`` frame.

    Example:
        >>> format_sse_event("token", {"content": "hi"})
        'data: {"type": "token", "content": "hi"}\\n\\n'
    """
    body = {"type": event_type, **payload}
    return f"data: {json.dumps(body)}\n\n"


def chunk_answer_text(answer: str, words_per_chunk: int = SSE_STREAM_WORD_CHUNK_SIZE) -> list[str]:
    """Split answer into word chunks for progressive SSE token events."""
    words = answer.split()
    if not words:
        return [answer] if answer else []
    chunks: list[str] = []
    for index in range(0, len(words), words_per_chunk):
        piece = " ".join(words[index : index + words_per_chunk])
        if index > 0:
            piece = f" {piece}"
        chunks.append(piece)
    return chunks


async def _yield_sse_frame(event_type: str, payload: dict[str, object]) -> AsyncIterator[str]:
    yield format_sse_event(event_type, payload)
    yield SSE_FLUSH_COMMENT
    await asyncio.sleep(0)
    if event_type == SSE_EVENT_TOKEN and SSE_STREAM_CHUNK_DELAY_SECONDS > 0:
        await asyncio.sleep(SSE_STREAM_CHUNK_DELAY_SECONDS)


async def stream_query_events(
    result: QueryResponse,
    *,
    graph_stages: list[str] | None = None,
    emit_started: bool = True,
) -> AsyncIterator[str]:
    """Yield SSE frames for graph status, answer tokens, and final payload."""
    if emit_started:
        async for frame in _yield_sse_frame(SSE_EVENT_STATUS, {"stage": SSE_STAGE_STARTED}):
            yield frame

    stages = graph_stages if graph_stages is not None else result.metadata.nodes_visited
    for node_name in stages:
        async for frame in _yield_sse_frame(SSE_EVENT_STATUS, {"stage": node_name}):
            yield frame

    for text_chunk in chunk_answer_text(result.answer):
        async for frame in _yield_sse_frame(SSE_EVENT_TOKEN, {"content": text_chunk}):
            yield frame

    async for frame in _yield_sse_frame(
        SSE_EVENT_DONE,
        {"result": result.model_dump(mode="json", by_alias=True)},
    ):
        yield frame


async def stream_error_event(detail: str, correlation_id: str | None = None) -> AsyncIterator[str]:
    """Yield a single SSE error frame."""
    error_body = ErrorResponse(detail=detail, correlation_id=correlation_id).model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    yield format_sse_event(SSE_EVENT_ERROR, error_body)
