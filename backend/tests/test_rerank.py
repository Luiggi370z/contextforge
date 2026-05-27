import uuid

from app.retrieval.models import RetrievedChunk
from app.retrieval.rerank import lexical_rerank, rerank_candidates


def _chunk(content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        score=score,
    )


def test_lexical_rerank_prefers_query_overlap():
    candidates = [
        _chunk("security password rotation policy", score=0.9),
        _chunk("paid time off PTO accrual days", score=0.5),
    ]
    ranked = lexical_rerank("PTO days", candidates, top_n=2)
    assert "PTO" in ranked[0].content


def test_rerank_candidates_respects_top_n():
    candidates = [_chunk(f"chunk {index}", score=float(index)) for index in range(10)]
    ranked = rerank_candidates("chunk", candidates, top_n=3)
    assert len(ranked) == 3
