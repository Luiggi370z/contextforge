from app.graph.builder import _after_route, build_agent_graph
from app.graph.state import GraphState


def test_build_agent_graph_compiles():
    graph = build_agent_graph()
    assert graph is not None


def test_after_route_direct():
    state: GraphState = {"route": "direct", "query": "hi"}
    assert _after_route(state) == "generate"


def test_after_route_rag():
    state: GraphState = {"route": "single_hop_rag", "query": "PTO?"}
    assert _after_route(state) == "retrieve"
