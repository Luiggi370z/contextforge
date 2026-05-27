import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.constants import GRAPH_NODE_RETRIEVE, ROUTE_DIRECT
from app.graph.builder import invoke_agent_graph
from app.retrieval.models import RetrievedChunk


@pytest.mark.asyncio
async def test_invoke_direct_route_skips_retrieve():
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "query": "hi",
            "route": ROUTE_DIRECT,
            "answer": "Hello",
            "nodes_visited": ["route", "generate", "validate_answer"],
            "documents": [],
            "citations": [],
            "abstained": False,
            "retrieval_scores": [],
        }
    )

    state = await invoke_agent_graph(
        "hi",
        db=MagicMock(),
        qdrant=MagicMock(),
        compiled_graph=mock_graph,
        thread_id="thread-direct",
    )

    assert state["route"] == ROUTE_DIRECT
    config = mock_graph.ainvoke.call_args.kwargs["config"]
    assert config["configurable"]["thread_id"] == "thread-direct"


@pytest.mark.asyncio
async def test_invoke_rag_flow_calls_hybrid_retrieve():
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Employees receive 15 PTO days per year.",
        score=0.9,
    )

    with pytest.MonkeyPatch.context() as patcher:
        hybrid_mock = AsyncMock(return_value=[chunk])
        patcher.setattr("app.graph.builder.hybrid_retrieve", hybrid_mock)

        state = await invoke_agent_graph(
            "How many PTO days?",
            db=MagicMock(),
            qdrant=MagicMock(),
            thread_id="thread-rag",
        )

    hybrid_mock.assert_awaited()
    assert GRAPH_NODE_RETRIEVE in state.get("nodes_visited", [])
    assert state.get("documents")
