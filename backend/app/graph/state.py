"""LangGraph state contract.

Fields prefixed with ``_`` are internal-only; ``app.graph.builder._finalize_graph_state``
strips them before the state leaves the graph and reaches the HTTP layer.
"""

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    query: str
    retrieval_query: str
    chat_history: list[dict[str, str]]
    route: str
    documents: list[dict]
    citations: list[dict]
    abstained: bool
    nodes_visited: list[str]
    answer: str
    retrieval_scores: list[float]
    # Per-run identifier propagated to every node log so traces are greppable.
    trace_id: str
    # Internal: full retrieved chunks (route -> grade -> generate share these).
    _chunks: list[Any]
    # Internal: chunks selected for the answer; validation re-checks against these.
    _selected_chunks: list[Any]
