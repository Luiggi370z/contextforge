"""Upload enqueues a job and returns a pollable job id; worker updates status."""

from __future__ import annotations

import uuid

import pytest

from app.core.constants import (
    INGESTION_JOB_COMPLETED,
    INGESTION_JOB_PROCESSING,
    INGESTION_JOB_QUEUED,
)


class _FakePool:
    def __init__(self) -> None:
        self.enqueued: list[tuple] = []

    async def enqueue_job(self, function, *args, **kwargs):  # noqa: ANN001
        self.enqueued.append((function, args, kwargs))

        class _Job:
            job_id = "fake-job"

        return _Job()


@pytest.mark.asyncio
async def test_enqueue_upload_creates_queued_job(monkeypatch):
    from app.api.v1.documents.service import DocumentService

    service = DocumentService()
    pool = _FakePool()

    captured = {}

    class _FakeRepo:
        async def create_job(self, session, *, filename):  # noqa: ANN001
            job = type(
                "J",
                (),
                {"id": uuid.uuid4(), "filename": filename, "status": INGESTION_JOB_QUEUED},
            )()
            captured["job"] = job
            return job

    service._repository = _FakeRepo()  # type: ignore[assignment]

    class _FakeSession:
        async def commit(self):
            return None

    job = await service.enqueue_upload(
        _FakeSession(),  # type: ignore[arg-type]
        pool,
        filename="policy.pdf",
        raw_bytes=b"%PDF-1.4 fake",
        content_type="application/pdf",
    )
    assert job.status == INGESTION_JOB_QUEUED
    assert pool.enqueued, "must enqueue an ARQ job"
    assert pool.enqueued[0][0] == "ingest_document_task"


def test_status_constants_are_distinct():
    values = {
        INGESTION_JOB_QUEUED,
        INGESTION_JOB_PROCESSING,
        INGESTION_JOB_COMPLETED,
    }
    assert len(values) == 3
