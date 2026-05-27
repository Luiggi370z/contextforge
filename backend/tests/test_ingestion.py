import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.constants import DOCUMENT_STATUS_INGESTED, DOCUMENT_STATUS_PROCESSING
from app.ingestion.service import ingest_document_text


@pytest.mark.asyncio
async def test_ingest_document_text_sets_status_and_upserts_qdrant():
    qdrant = AsyncMock()
    qdrant.upsert_chunks = AsyncMock()

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()

    document = MagicMock()
    document.id = uuid.uuid4()
    document.status = DOCUMENT_STATUS_PROCESSING

    with patch("app.ingestion.service.Document", return_value=document):
        with patch("app.ingestion.service.Chunk") as chunk_cls:
            chunk_cls.side_effect = lambda **kwargs: MagicMock(id=uuid.uuid4(), **kwargs)
            with patch("app.ingestion.service.split_text", return_value=["part-a", "part-b"]):
                result = await ingest_document_text(
                    session,
                    qdrant,
                    filename="policy.md",
                    content="# Policy\n\nBody text.",
                )

    assert result.status == DOCUMENT_STATUS_INGESTED
    qdrant.upsert_chunks.assert_awaited_once()
    upsert_args = qdrant.upsert_chunks.await_args.args
    assert len(upsert_args[1]) == 2
    assert upsert_args[2] == ["part-a", "part-b"]
    session.commit.assert_awaited()
