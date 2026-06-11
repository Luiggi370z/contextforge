"""Ingest a document: split, persist chunks (with metadata), upsert vectors."""

from __future__ import annotations

import uuid

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import (
    DEFAULT_MARKDOWN_CONTENT_TYPE,
    DOCUMENT_STATUS_INGESTED,
    DOCUMENT_STATUS_PROCESSING,
)
from app.db.models import Chunk, Document
from app.ingestion.chunker import split_text_into_chunks
from app.retrieval.qdrant_store import QdrantStore

log = structlog.get_logger(__name__)


async def ingest_document_text(
    db: AsyncSession,
    qdrant: QdrantStore,
    *,
    filename: str,
    content: str,
    content_type: str = DEFAULT_MARKDOWN_CONTENT_TYPE,
) -> Document:
    """Persist a document + chunks + vectors with structure-aware contextual prefixes."""
    doc = Document(
        filename=filename,
        content_type=content_type,
        status=DOCUMENT_STATUS_PROCESSING,
    )
    db.add(doc)
    await db.flush()

    pieces = split_text_into_chunks(content, filename=filename)
    chunk_ids: list[uuid.UUID] = []
    embedding_texts: list[str] = []
    bodies: list[str] = []
    for piece in pieces:
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=piece.metadata["chunk_index"],
            content=piece.body,
            metadata_={
                **piece.metadata,
                "context_prefix": piece.context_prefix,
            },
        )
        db.add(chunk)
        await db.flush()
        chunk_ids.append(chunk.id)
        embedding_texts.append(piece.content)
        bodies.append(piece.body)
        chunk.qdrant_point_id = str(chunk.id)

    await qdrant.upsert_chunks(doc.id, chunk_ids, embedding_texts, bodies=bodies)
    doc.status = DOCUMENT_STATUS_INGESTED
    await db.commit()
    await db.refresh(doc)
    log.info("ingest_complete", document_id=str(doc.id), chunks=len(pieces))
    return doc
