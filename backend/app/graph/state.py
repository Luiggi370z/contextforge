from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    query: str
    route: str
    documents: list[dict]
    citations: list[dict]
    abstained: bool
    nodes_visited: list[str]
    answer: str
    retrieval_scores: list[float]
