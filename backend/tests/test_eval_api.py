"""Eval service runs the golden set across configs and computes IR metrics."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
async def test_eval_service_computes_metrics_per_config(real_embeddings):
    """EvalService.run produces one metrics dict per requested config."""
    from app.api.v1.eval.service import EvalService

    captured = []

    class _FakeRepo:
        async def create_run(self, session, *, label, config, metrics, row_count):  # noqa: ANN001
            run = type(
                "R",
                (),
                {
                    "id": label,
                    "label": label,
                    "config": config,
                    "metrics": metrics,
                    "row_count": row_count,
                },
            )()
            captured.append(run)
            return run

        async def add_result(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

    class _FakeSession:
        async def commit(self):
            return None

    service = EvalService(repository=_FakeRepo())  # type: ignore[arg-type]
    runs = await service.run(
        _FakeSession(),  # type: ignore[arg-type]
        configs=["dense_only", "hybrid", "hybrid_rerank"],
    )
    assert len(runs) == 3
    for run in runs:
        assert "recall@5" in run.metrics
        assert run.metrics["recall@5"] >= 0.0
