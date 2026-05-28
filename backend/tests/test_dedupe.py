import uuid

from app.retrieval.dedupe import dedupe_chunks_by_content, dedupe_citations
from app.retrieval.models import RetrievedChunk


def test_dedupe_chunks_by_content_keeps_first_ranked():
    content = "Access to production systems requires MFA."
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content=content,
            score=0.9,
            relevance_score=0.403,
        ),
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content=content,
            score=0.8,
            relevance_score=0.403,
        ),
    ]
    deduped = dedupe_chunks_by_content(chunks)
    assert len(deduped) == 1
    assert deduped[0].chunk_id == chunks[0].chunk_id


def test_dedupe_citations_by_snippet():
    snippet = "Access to production systems requires MFA."
    citations = [
        {"chunk_id": str(uuid.uuid4()), "snippet": snippet, "score": 0.403},
        {"chunk_id": str(uuid.uuid4()), "snippet": snippet, "score": 0.403},
    ]
    assert len(dedupe_citations(citations)) == 1
