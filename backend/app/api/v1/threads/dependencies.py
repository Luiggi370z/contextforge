"""FastAPI dependencies for thread routes."""

from functools import lru_cache

from app.api.v1.threads.service import ThreadService


@lru_cache
def get_thread_service() -> ThreadService:
    """Singleton thread service for request handlers."""
    # TODO(session-in-service-di): see documents/dependencies.py
    return ThreadService()
