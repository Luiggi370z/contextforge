"""LangGraph workflow assembly."""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from app.graph.state import GraphState

log = structlog.get_logger(__name__)


def build_agent_graph() -> Any:
    """Build graph structure; execution uses run_agent_pipeline with DB deps."""
    graph = StateGraph(GraphState)
    graph.add_node("route", lambda s: s)
    graph.add_edge(START, "route")
    graph.add_edge("route", END)
    return graph.compile()
