"""FastAPI dependencies for query routes."""

from functools import lru_cache

from app.api.v1.query.service import QueryService


@lru_cache
def get_query_service() -> QueryService:
    """Singleton query service for request handlers."""
    return QueryService()
