from __future__ import annotations

import uuid

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.ingestion.chunker import split_text
from app.retrieval.qdrant_store import QdrantStore

log = structlog.get_logger(__name__)


async def ingest_document_text(
    db: AsyncSession,
    qdrant: QdrantStore,
    *,
    filename: str,
    content: str,
    content_type: str = "text/markdown",
) -> Document:
    doc = Document(filename=filename, content_type=content_type, status="processing")
    db.add(doc)
    await db.flush()

    pieces = split_text(content)
    chunk_ids: list[uuid.UUID] = []
    texts: list[str] = []
    for index, piece in enumerate(pieces):
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=index,
            content=piece,
            metadata_={"filename": filename},
        )
        db.add(chunk)
        await db.flush()
        chunk_ids.append(chunk.id)
        texts.append(piece)
        chunk.qdrant_point_id = str(chunk.id)

    await qdrant.upsert_chunks(doc.id, chunk_ids, texts)
    doc.status = "ingested"
    await db.commit()
    await db.refresh(doc)
    log.info("ingest_complete", document_id=str(doc.id), chunks=len(pieces))
    return doc
