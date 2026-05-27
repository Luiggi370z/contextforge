"""Query execution (agent graph orchestration)."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.query.schemas import QueryRequest, QueryResponse
from app.graph.runner import run_query


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
