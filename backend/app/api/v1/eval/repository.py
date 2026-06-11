from __future__ import annotations

import uuid

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.eval.models import EvalResult, EvalRun


class EvalRepository:
    async def create_run(
        self, session: AsyncSession, *, label: str, config: dict, metrics: dict, row_count: int
    ) -> EvalRun:
        run = EvalRun(label=label, config=config, metrics=metrics, row_count=row_count)
        session.add(run)
        await session.flush()
        return run

    async def add_result(
        self,
        session: AsyncSession,
        *,
        run_id: uuid.UUID,
        question: str,
        reference_doc: str,
        retrieved_docs: list[str],
        abstained: bool,
    ) -> None:
        session.add(
            EvalResult(
                run_id=run_id,
                question=question,
                reference_doc=reference_doc,
                retrieved_docs=retrieved_docs,
                abstained=abstained,
            )
        )

    async def list_runs(self, session: AsyncSession, limit: int = 50) -> list[EvalRun]:
        result = await session.execute(
            select(EvalRun).order_by(desc(EvalRun.created_at)).limit(limit)
        )
        return list(result.scalars().all())

    async def count_runs(self, session: AsyncSession) -> int:
        total = await session.scalar(select(func.count()).select_from(EvalRun))
        return int(total or 0)
