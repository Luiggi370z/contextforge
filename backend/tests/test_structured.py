import uuid

from app.llm.structured import _heuristic_route, grade_retrieval
from app.retrieval.hybrid import RetrievedChunk


def test_heuristic_route_direct():
    d = _heuristic_route("hi")
    assert d.route == "direct"


def test_heuristic_route_rag():
    d = _heuristic_route("How many PTO days do employees get per year?")
    assert d.route == "single_hop_rag"


def test_heuristic_route_multi_hop():
    d = _heuristic_route("First compare PTO then explain remote work steps")
    assert d.route == "multi_hop"


def test_grade_abstain_on_empty():
    g = grade_retrieval([], "query")
    assert g.should_abstain is True


def test_grade_passes_with_chunks():
    chunks = [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="PTO policy text",
            score=0.9,
        )
    ]
    g = grade_retrieval(chunks, "PTO")
    assert g.should_abstain is False
