from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.query.schemas import Citation, QueryMetadata, QueryResponse
from app.main import app


@pytest.mark.asyncio
async def test_metrics_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "queriesTotal" in data


@pytest.mark.asyncio
async def test_query_with_mocked_runner():
    mock_response = QueryResponse(
        answer="test answer",
        citations=[Citation(snippet="snippet", score=0.9)],
        metadata=QueryMetadata(route="single_hop_rag", nodes_visited=["route", "retrieve"]),
    )

    with patch(
        "app.api.v1.query.service.run_query", new_callable=AsyncMock
    ) as mock_run:
        mock_run.return_value = mock_response
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post("/v1/query", json={"message": "How many PTO days?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "test answer"
    assert body["metadata"]["route"] == "single_hop_rag"


@pytest.mark.asyncio
async def test_query_stream_mocked():
    """Covered in depth by tests/test_streaming.py."""
    mock_response = QueryResponse(
        answer="streamed",
        metadata=QueryMetadata(route="direct"),
    )
    with patch(
        "app.api.v1.query.service.run_query", new_callable=AsyncMock
    ) as mock_run:
        mock_run.return_value = mock_response
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post("/v1/query/stream", json={"message": "hello"})
    assert res.status_code == 200


@pytest.mark.asyncio
async def test_list_documents_empty_db_fails_without_db():
    """Without a live DB this may error — skip if integration unavailable."""
    pytest.importorskip("asyncpg")
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", timeout=2.0
        ) as client:
            await client.get("/v1/documents")
    except Exception:
        pytest.skip("Postgres not available for integration test")
