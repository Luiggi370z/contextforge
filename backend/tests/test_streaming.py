import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.query.schemas import Citation, QueryMetadata, QueryResponse
from app.api.v1.query.streaming import chunk_answer_text, format_sse_event, stream_query_events
from app.core.constants import SSE_EVENT_DONE, SSE_EVENT_STATUS, SSE_EVENT_TOKEN
from app.main import app


def _parse_sse_data_frames(frames: list[str]) -> list[dict]:
    events: list[dict] = []
    for frame in frames:
        if not frame.startswith("data: "):
            continue
        events.append(json.loads(frame.removeprefix("data: ").strip()))
    return events


def test_format_sse_event():
    frame = format_sse_event("token", {"content": "hello"})
    assert frame.startswith("data: ")
    payload = json.loads(frame.removeprefix("data: ").strip())
    assert payload["type"] == "token"
    assert payload["content"] == "hello"


def test_chunk_answer_text_splits_words():
    chunks = chunk_answer_text("one two three four five six", words_per_chunk=2)
    assert len(chunks) == 3
    assert "".join(chunks) == "one two three four five six"


@pytest.mark.asyncio
async def test_stream_query_events_order():
    result = QueryResponse(
        answer="alpha beta gamma",
        thread_id=uuid.uuid4(),
        citations=[Citation(snippet="src", score=0.5)],
        metadata=QueryMetadata(route="single_hop_rag", nodes_visited=["route", "retrieve"]),
    )
    frames = [frame async for frame in stream_query_events(result)]
    events = _parse_sse_data_frames(frames)
    types = [event["type"] for event in events]
    assert types[0] == SSE_EVENT_STATUS
    assert SSE_EVENT_TOKEN in types
    assert types[-1] == SSE_EVENT_DONE
    done_payload = events[-1]
    assert done_payload["result"]["citations"][0]["snippet"] == "src"


@pytest.mark.asyncio
async def test_query_stream_returns_multiple_sse_events():
    thread_id = uuid.uuid4()
    mock_response = QueryResponse(
        answer="one two three four",
        thread_id=thread_id,
        metadata=QueryMetadata(route="direct", nodes_visited=["route"]),
    )
    async def fake_stream_graph(*_args, **_kwargs):
        yield "route"
        yield mock_response

    with patch("app.api.v1.query.service.stream_query_graph", fake_stream_graph):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post("/v1/query/stream", json={"message": "hello"})

    assert res.status_code == 200
    assert "text/event-stream" in res.headers.get("content-type", "")
    assert res.headers.get("cache-control") == "no-cache"

    events = []
    for block in res.text.split("\n\n"):
        if not block.startswith("data: "):
            continue
        events.append(json.loads(block.removeprefix("data: ").strip()))

    token_events = [event for event in events if event["type"] == SSE_EVENT_TOKEN]
    done_events = [event for event in events if event["type"] == SSE_EVENT_DONE]
    assert len(token_events) >= 1
    assert len(done_events) == 1
    assert done_events[0]["result"]["threadId"] == str(thread_id)
