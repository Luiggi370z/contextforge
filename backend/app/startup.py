"""Application startup and shutdown lifecycle."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.graph.checkpointer import postgres_checkpointer

log = structlog.get_logger(__name__)


@asynccontextmanager
async def application_lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Configure logging, LangGraph checkpointer, and shared app state.

    Example:
        Used as ``lifespan=application_lifespan`` on the FastAPI constructor.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("application_starting", app_env=settings.app_env)

    async with postgres_checkpointer() as checkpointer:
        application.state.checkpointer = checkpointer
        if checkpointer is not None:
            log.info("langgraph_checkpointer_attached")
        else:
            log.warning("langgraph_checkpointer_unavailable")
        yield

    log.info("application_shutdown")
