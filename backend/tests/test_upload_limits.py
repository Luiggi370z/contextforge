import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.documents.dependencies import get_arq_pool, get_document_service, get_qdrant
from app.db.session import get_db
from app.main import app


async def _fake_db():
    yield MagicMock()


def _clear_overrides() -> None:
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_document_service, None)
    app.dependency_overrides.pop(get_qdrant, None)
    app.dependency_overrides.pop(get_arq_pool, None)


@pytest.fixture(autouse=True)
def _settings_cache():
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    _clear_overrides()


@pytest.mark.asyncio
async def test_upload_rejects_file_over_configured_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "4")
    from app.core.config import get_settings

    get_settings.cache_clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/documents/upload",
            files={"file": ("large.txt", b"12345", "text/plain")},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Upload exceeds maximum allowed size"


@pytest.mark.asyncio
async def test_text_ingest_rejects_content_over_configured_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "4")
    from app.core.config import get_settings

    get_settings.cache_clear()
    service = MagicMock()
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[get_document_service] = lambda: service
    app.dependency_overrides[get_qdrant] = lambda: MagicMock()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/documents",
            json={"filename": "large.md", "content": "12345"},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Upload exceeds maximum allowed size"
    service.ingest_text.assert_not_called()


@pytest.mark.asyncio
async def test_small_upload_still_enqueues(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1000000")
    from app.core.config import get_settings

    get_settings.cache_clear()

    class FakeService:
        async def enqueue_upload(self, *_args, **_kwargs):
            return SimpleNamespace(
                id=uuid.uuid4(),
                filename="small.txt",
                status="queued",
                progress=0,
                document_id=None,
                error=None,
            )

    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[get_document_service] = lambda: FakeService()
    app.dependency_overrides[get_arq_pool] = lambda: MagicMock()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/documents/upload",
            files={"file": ("small.txt", b"ok", "text/plain")},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
