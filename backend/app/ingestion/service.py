"""Ingest a document: split, persist chunks (with metadata), upsert vectors."""

from __future__ import annotations

import asyncio
import uuid

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.constants import (
    DEFAULT_MARKDOWN_CONTENT_TYPE,
    DOCUMENT_STATUS_INGESTED,
    DOCUMENT_STATUS_PROCESSING,
)
from app.db.models import Chunk, Document
from app.ingestion.chunker import ChunkPiece, split_blocks_into_chunks
from app.ingestion.loaders import LoadedBlock, load_document
from app.llm.providers import LLMProvider, get_llm_provider
from app.retrieval.qdrant_store import QdrantStore

log = structlog.get_logger(__name__)


async def _resolve_context_prefix(
    piece: ChunkPiece,
    *,
    provider: LLMProvider | None,
    filename: str,
    full_document: str,
) -> str:
    """Return the prefix to prepend before embedding a chunk.

    Without a provider, this is the chunker's static structural prefix. With one
    (contextual retrieval enabled), it is an LLM-written situating blurb. The
    heuristic provider returns the same static prefix, keeping the eval gate
    byte-identical.

    Example:
        >>> import asyncio
        >>> from app.ingestion.chunker import ChunkPiece
        >>> prefix = "Document: pto.md > Section: PTO"
        >>> piece = ChunkPiece(body="20 days", context_prefix=prefix, metadata={"section": "PTO"})
        >>> asyncio.run(
        ...     _resolve_context_prefix(
        ...         piece, provider=None, filename="pto.md", full_document="..."
        ...     )
        ... )
        'Document: pto.md > Section: PTO'
    """
    if provider is None:
        return piece.context_prefix
    return await provider.contextualize(
        document_title=filename,
        section=piece.metadata.get("section") or "Body",
        chunk=piece.body,
        full_document=full_document,
    )


async def ingest_document_blocks(
    db: AsyncSession,
    qdrant: QdrantStore,
    *,
    filename: str,
    blocks: list[LoadedBlock],
    content_type: str,
) -> Document:
    """Persist a document + chunks + vectors from pre-loaded blocks.

    Carries each block's page/section through the chunker into the new
    ``chunks.page`` / ``chunks.section`` columns.
    """
    settings = get_settings()
    provider = get_llm_provider() if settings.contextual_retrieval_enabled else None
    full_document = "\n\n".join(block.text for block in blocks) if provider is not None else ""

    doc = Document(
        filename=filename,
        content_type=content_type,
        status=DOCUMENT_STATUS_PROCESSING,
    )
    db.add(doc)
    await db.flush()

    pieces = await asyncio.to_thread(split_blocks_into_chunks, blocks, filename=filename)
    chunk_ids: list[uuid.UUID] = []
    embedding_texts: list[str] = []
    bodies: list[str] = []
    for piece in pieces:
        context_prefix = await _resolve_context_prefix(
            piece, provider=provider, filename=filename, full_document=full_document
        )
        embedded_text = f"{context_prefix}\n\n{piece.body}" if context_prefix else piece.body
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=piece.metadata["chunk_index"],
            content=piece.body,
            section=piece.metadata.get("section"),
            page=piece.metadata.get("page"),
            metadata_={
                **piece.metadata,
                "context_prefix": context_prefix,
            },
        )
        db.add(chunk)
        await db.flush()
        chunk_ids.append(chunk.id)
        embedding_texts.append(embedded_text)
        bodies.append(piece.body)
        chunk.qdrant_point_id = str(chunk.id)

    await qdrant.upsert_chunks(doc.id, chunk_ids, embedding_texts, bodies=bodies)
    doc.status = DOCUMENT_STATUS_INGESTED
    await db.commit()
    await db.refresh(doc)
    log.info("ingest_complete", document_id=str(doc.id), chunks=len(pieces))
    return doc


async def ingest_document_text(
    db: AsyncSession,
    qdrant: QdrantStore,
    *,
    filename: str,
    content: str,
    content_type: str = DEFAULT_MARKDOWN_CONTENT_TYPE,
) -> Document:
    """Back-compat text ingest: wrap content in markdown-derived blocks."""
    blocks = await asyncio.to_thread(
        load_document,
        filename=filename,
        raw_bytes=content.encode("utf-8"),
        content_type=content_type,
    )
    return await ingest_document_blocks(
        db, qdrant, filename=filename, blocks=blocks, content_type=content_type
    )
