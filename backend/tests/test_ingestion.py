import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.constants import DOCUMENT_STATUS_INGESTED, DOCUMENT_STATUS_PROCESSING
from app.ingestion.chunker import ChunkPiece
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

    pieces = [
        ChunkPiece(
            body="part-a body",
            context_prefix="Document: policy.md > Section: Policy",
            metadata={"filename": "policy.md", "section": "Policy", "chunk_index": 0},
        ),
        ChunkPiece(
            body="part-b body",
            context_prefix="Document: policy.md > Section: Policy",
            metadata={"filename": "policy.md", "section": "Policy", "chunk_index": 1},
        ),
    ]

    with patch("app.ingestion.service.Document", return_value=document):
        with patch("app.ingestion.service.Chunk") as chunk_cls:
            chunk_cls.side_effect = lambda **kwargs: MagicMock(id=uuid.uuid4(), **kwargs)
            with patch("app.ingestion.service.split_blocks_into_chunks", return_value=pieces):
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
    # Embedded text is body + context prefix, not just the body, so dense retrieval
    # benefits from section/file context (Anthropic-style contextual chunks).
    embedded_texts = upsert_args[2]
    assert embedded_texts[0].endswith("part-a body")
    assert "Section: Policy" in embedded_texts[0]
    # The clean bodies must flow as the ``bodies`` kwarg so citations never leak
    # the structural prefix that the embedding text carries.
    upsert_kwargs = qdrant.upsert_chunks.await_args.kwargs
    assert "bodies" in upsert_kwargs
    assert all(not b.startswith("Document: ") for b in upsert_kwargs["bodies"])
    session.commit.assert_awaited()
