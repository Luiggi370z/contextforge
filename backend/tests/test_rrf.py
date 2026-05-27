from app.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_merges_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    ids = [doc_id for doc_id, _ in fused]
    assert "b" in ids
    assert ids[0] == "b"


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([]) == []
