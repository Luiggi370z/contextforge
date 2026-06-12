"""Document preview content: service fallback logic + HTTP route."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.documents.dependencies import get_document_service
from app.api.v1.documents.models import Document
from app.api.v1.documents.service import DocumentService
from app.db.session import get_db
from app.main import app


def _service_with(document: Document | None) -> DocumentService:
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=document)
    return DocumentService(repository=repository)


@pytest.mark.asyncio
async def test_content_returns_raw_bytes_with_original_media_type():
    document = Document(
        id=uuid.uuid4(),
        filename="policy.pdf",
        content_type="application/pdf",
        raw_bytes=b"%PDF-1.4 fake",
        size_bytes=13,
    )
    service = _service_with(document)

    result = await service.get_document_content(MagicMock(), document.id)

    assert result == (b"%PDF-1.4 fake", "application/pdf")


@pytest.mark.asyncio
async def test_content_returns_none_for_missing_document():
    service = _service_with(None)
    assert await service.get_document_content(MagicMock(), uuid.uuid4()) is None


@pytest.mark.asyncio
async def test_content_route_serves_bytes_and_404():
    document = Document(
        id=uuid.uuid4(),
        filename="notes.txt",
        content_type="text/plain",
        raw_bytes=b"hello world",
        size_bytes=11,
    )
    service = _service_with(document)

    async def _fake_db():
        yield MagicMock()

    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[get_document_service] = lambda: service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            ok = await client.get(f"/v1/documents/{document.id}/content")
            assert ok.status_code == 200
            assert ok.content == b"hello world"
            assert ok.headers["content-type"].startswith("text/plain")

            service._repository.get_by_id = AsyncMock(return_value=None)  # type: ignore[attr-defined]
            missing = await client.get(f"/v1/documents/{uuid.uuid4()}/content")
            assert missing.status_code == 404
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_document_service, None)
