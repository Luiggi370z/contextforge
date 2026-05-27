"""Query execution (agent graph orchestration)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.query.schemas import QueryRequest, QueryResponse
from app.api.v1.query.streaming import stream_error_event, stream_query_events
from app.core.constants import ERROR_INTERNAL_SERVER
from app.core.exceptions import AppException
from app.graph.runner import run_query

log = structlog.get_logger(__name__)


class QueryService:
    """Runs agentic RAG queries and persists thread messages."""

    async def execute(
        self,
        body: QueryRequest,
        session: AsyncSession,
        *,
        checkpointer: Any | None = None,
        compiled_graph: Any | None = None,
    ) -> QueryResponse:
        """Run a full query through the LangGraph pipeline."""
        return await run_query(
            body,
            session,
            checkpointer=checkpointer,
            compiled_graph=compiled_graph,
        )

    async def stream_execute(
        self,
        body: QueryRequest,
        session: AsyncSession,
        *,
        checkpointer: Any | None = None,
        compiled_graph: Any | None = None,
    ) -> AsyncIterator[str]:
        """Run a query and yield SSE frames (status, tokens, done)."""
        try:
            result = await self.execute(
                body,
                session,
                checkpointer=checkpointer,
                compiled_graph=compiled_graph,
            )
        except AppException as error:
            async for frame in stream_error_event(error.detail, error.correlation_id):
                yield frame
            return
        except Exception as error:
            log.exception("stream_query_failed", error=str(error))
            async for frame in stream_error_event(ERROR_INTERNAL_SERVER):
                yield frame
            return

        async for frame in stream_query_events(result):
            yield frame
