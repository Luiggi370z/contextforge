import uuid

from app.api.v1.health.schemas import HealthResponse
from app.api.v1.query.schemas import QueryMetadata, QueryResponse
from app.schemas.errors import ErrorResponse


def test_health_response_serializes_camel_case():
    payload = HealthResponse(status="ok", app_env="test").model_dump(
        mode="json", by_alias=True
    )
    assert payload == {"status": "ok", "appEnv": "test"}


def test_query_response_serializes_camel_case():
    payload = QueryResponse(
        answer="hello",
        thread_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        metadata=QueryMetadata(
            route="single_hop_rag",
            nodes_visited=["route"],
            graph_checkpoint_enabled=True,
        ),
    ).model_dump(mode="json", by_alias=True)
    assert payload["threadId"] == "00000000-0000-0000-0000-000000000001"
    assert payload["metadata"]["nodesVisited"] == ["route"]
    assert payload["metadata"]["graphCheckpointEnabled"] is True


def test_query_request_accepts_camel_case():
    from app.api.v1.query.schemas import QueryRequest

    body = QueryRequest.model_validate(
        {
            "message": "hi",
            "threadId": "00000000-0000-0000-0000-000000000002",
        }
    )
    assert body.thread_id == uuid.UUID("00000000-0000-0000-0000-000000000002")


def test_error_response_serializes_camel_case():
    payload = ErrorResponse(detail="bad", correlation_id="corr-1").model_dump(
        mode="json", by_alias=True
    )
    assert payload["correlationId"] == "corr-1"
