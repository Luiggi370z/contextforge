"""Live pipeline stage events: emit_stage, graph custom stream, SSE payloads."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.query.schemas import QueryMetadata, QueryResponse
from app.core.constants import (
    GRAPH_NODE_GENERATE,
    GRAPH_NODE_GRADE,
    GRAPH_NODE_RETRIEVE,
    GRAPH_NODE_ROUTE,
    GRAPH_NODE_VALIDATE,
    SSE_EVENT_STATUS,
    STAGE_GENERATE_LLM,
    STAGE_GRADE,
    STAGE_PHASE_END,
    STAGE_PHASE_START,
    STAGE_RETRIEVE_RERANK,
    STAGE_RETRIEVE_SEARCH,
    STAGE_ROUTE,
    STAGE_VALIDATE,
)
from app.graph.builder import stream_invoke_agent_graph
from app.graph.progress import StageEvent, emit_stage
from app.main import app
from app.retrieval.models import RetrievedChunk


def test_emit_stage_is_noop_outside_graph_context():
    # No graph run in scope: must not raise, must not require a writer.
    emit_stage(STAGE_RETRIEVE_RERANK, candidates=5)


def test_stage_event_payload_shape():
    assert StageEvent(stage=STAGE_ROUTE).to_payload() == {
        "stage": STAGE_ROUTE,
        "phase": STAGE_PHASE_START,
    }
    payload = StageEvent(
        stage=STAGE_GENERATE_LLM, detail={"provider": "ollama"}
    ).to_payload()
    assert payload["detail"] == {"provider": "ollama"}


@pytest.mark.asyncio
async def test_stream_emits_substages_in_pipeline_order():
    """Run the real graph and assert start + node-complete events interleave."""
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Employees receive 15 PTO days per year.",
        score=0.9,
    )

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr("app.graph.builder.hybrid_retrieve", AsyncMock(return_value=[chunk]))

        items = [
            item
            async for item in stream_invoke_agent_graph(
                "How many PTO days?",
                db=MagicMock(),
                qdrant=MagicMock(),
                thread_id="thread-progress",
            )
        ]

    events = [item for item in items if isinstance(item, StageEvent)]
    final_state = items[-1]
    assert not isinstance(final_state, StageEvent)
    assert final_state.get("answer")

    starts = [event.stage for event in events if event.phase == STAGE_PHASE_START]
    ends = [event.stage for event in events if event.phase == STAGE_PHASE_END]

    # Sub-stage starts in pipeline order (hybrid_retrieve is patched, so no
    # retrieve.rerank here; that emission lives inside the real retriever).
    assert starts == [
        STAGE_ROUTE,
        STAGE_RETRIEVE_SEARCH,
        STAGE_GRADE,
        STAGE_GENERATE_LLM,
        STAGE_VALIDATE,
    ]
    # Node completions still stream via the updates channel.
    assert ends == [
        GRAPH_NODE_ROUTE,
        GRAPH_NODE_RETRIEVE,
        GRAPH_NODE_GRADE,
        GRAPH_NODE_GENERATE,
        GRAPH_NODE_VALIDATE,
    ]
    # generate.llm carries the active provider.
    generate_event = next(event for event in events if event.stage == STAGE_GENERATE_LLM)
    assert (generate_event.detail or {}).get("provider") == "heuristic"


@pytest.mark.asyncio
async def test_direct_route_skips_retrieval_stages():
    items = [
        item
        async for item in stream_invoke_agent_graph(
            "hi",
            db=MagicMock(),
            qdrant=MagicMock(),
            thread_id="thread-direct",
        )
    ]
    starts = [
        item.stage
        for item in items
        if isinstance(item, StageEvent) and item.phase == STAGE_PHASE_START
    ]
    assert STAGE_RETRIEVE_SEARCH not in starts
    assert STAGE_GRADE not in starts


@pytest.mark.asyncio
async def test_sse_status_events_carry_stage_phase_and_detail():
    thread_id = uuid.uuid4()
    mock_response = QueryResponse(
        answer="answer",
        thread_id=thread_id,
        metadata=QueryMetadata(route="single_hop_rag", nodes_visited=["route"]),
    )

    async def fake_stream_graph(*_args, **_kwargs):
        yield StageEvent(stage=STAGE_ROUTE)
        yield StageEvent(stage=STAGE_GENERATE_LLM, detail={"provider": "ollama"})
        yield StageEvent(stage=GRAPH_NODE_GENERATE, phase=STAGE_PHASE_END)
        yield mock_response

    with patch("app.api.v1.query.service.stream_query_graph", fake_stream_graph):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post("/v1/query/stream", json={"message": "hello"})

    assert res.status_code == 200
    events = []
    for block in res.text.split("\n\n"):
        if block.startswith("data: "):
            events.append(json.loads(block.removeprefix("data: ").strip()))

    status_events = [event for event in events if event["type"] == SSE_EVENT_STATUS]
    by_stage = {event["stage"]: event for event in status_events}
    assert by_stage[STAGE_ROUTE]["phase"] == STAGE_PHASE_START
    assert by_stage[STAGE_GENERATE_LLM]["detail"] == {"provider": "ollama"}
    assert by_stage[GRAPH_NODE_GENERATE]["phase"] == STAGE_PHASE_END
