"""SSE formatting and answer chunking for query streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.api.v1.query.schemas import QueryResponse
from app.core.constants import (
    SSE_EVENT_DONE,
    SSE_EVENT_ERROR,
    SSE_EVENT_STATUS,
    SSE_EVENT_TOKEN,
    SSE_STAGE_STARTED,
    SSE_STREAM_WORD_CHUNK_SIZE,
)


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


async def stream_query_events(result: QueryResponse) -> AsyncIterator[str]:
    """Yield SSE frames for graph status, answer tokens, and final payload."""
    yield format_sse_event(SSE_EVENT_STATUS, {"stage": SSE_STAGE_STARTED})

    for node_name in result.metadata.nodes_visited:
        yield format_sse_event(SSE_EVENT_STATUS, {"stage": node_name})

    for text_chunk in chunk_answer_text(result.answer):
        yield format_sse_event(SSE_EVENT_TOKEN, {"content": text_chunk})

    yield format_sse_event(
        SSE_EVENT_DONE,
        {"result": result.model_dump(mode="json")},
    )


async def stream_error_event(detail: str, correlation_id: str | None = None) -> AsyncIterator[str]:
    """Yield a single SSE error frame."""
    payload: dict[str, object] = {"detail": detail}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    yield format_sse_event(SSE_EVENT_ERROR, payload)
