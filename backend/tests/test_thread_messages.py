import uuid
from datetime import UTC, datetime

from app.api.v1.threads.models import Message
from app.api.v1.threads.schemas import message_response_from_orm


def test_message_response_from_orm_restores_citations():
    message = Message(
        id=uuid.uuid4(),
        thread_id=uuid.uuid4(),
        role="assistant",
        content="MFA is required for production systems.",
        metadata_={
            "route": "single_hop_rag",
            "abstained": False,
            "nodes_visited": ["route", "retrieve"],
            "retrieval_scores": [0.4],
            "graph_checkpoint_enabled": False,
            "citations": [
                {
                    "snippet": "Access to production systems requires MFA.",
                    "score": 0.403,
                }
            ],
        },
        created_at=datetime.now(UTC),
    )
    response = message_response_from_orm(message)
    assert response.metadata is not None
    assert response.metadata.route == "single_hop_rag"
    assert len(response.citations) == 1
    assert response.citations[0].snippet.startswith("Access to production")
