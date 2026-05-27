"""FastAPI dependencies for query routes."""

from functools import lru_cache
from typing import Any

from fastapi import Request

from app.api.v1.query.service import QueryService


@lru_cache
def get_query_service() -> QueryService:
    """Singleton query service for request handlers."""
    # TODO(session-in-service-di): see documents/dependencies.py
    return QueryService()


def get_langgraph_checkpointer(request: Request) -> Any | None:
    """Postgres checkpointer from app lifespan (None if unavailable)."""
    return getattr(request.app.state, "checkpointer", None)


def get_compiled_agent_graph(request: Request) -> Any | None:
    """Compiled LangGraph agent from app lifespan."""
    return getattr(request.app.state, "agent_graph", None)
