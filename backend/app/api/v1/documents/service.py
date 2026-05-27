"""Document ingestion and listing business logic."""

from __future__ import annotations

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.documents.models import DocumentListResult
from app.api.v1.documents.repository import DocumentRepository
from app.core.constants import (
    DEFAULT_TEXT_CONTENT_TYPE,
    DEFAULT_UPLOAD_FILENAME,
)
from app.core.exceptions import IngestionError
from app.db.models import Document
from app.ingestion.service import ingest_document_text
from app.retrieval.qdrant_store import QdrantStore

log = structlog.get_logger(__name__)


class DocumentService:
    """Coordinates repository, Qdrant, and ingestion pipeline."""

    def __init__(
        self,
        repository: DocumentRepository | None = None,
    ) -> None:
        self._repository = repository or DocumentRepository()

    async def list_documents(self, session: AsyncSession) -> DocumentListResult:
        """List recent documents with total count."""
        items = await self._repository.list_recent(session)
        total = await self._repository.count(session)
        return DocumentListResult(items=items, total=total)

    async def ingest_text(
        self,
        session: AsyncSession,
        qdrant: QdrantStore,
        *,
        filename: str,
        content: str,
        content_type: str = DEFAULT_TEXT_CONTENT_TYPE,
    ) -> Document:
        """Ingest plain text as a document (chunk, embed, upsert)."""
        try:
            document = await ingest_document_text(
                session,
                qdrant,
                filename=filename,
                content=content,
                content_type=content_type,
            )
            log.info("document_ingested", document_id=str(document.id), filename=filename)
            return document
        except Exception as error:
            log.exception("ingest_failed", filename=filename, error=str(error))
            raise IngestionError(str(error)) from error

    async def ingest_upload(
        self,
        session: AsyncSession,
        qdrant: QdrantStore,
        *,
        filename: str | None,
        raw_bytes: bytes,
        content_type: str | None,
    ) -> Document:
        """Decode uploaded bytes and ingest as a document."""
        text = raw_bytes.decode("utf-8", errors="replace")
        resolved_name = filename or DEFAULT_UPLOAD_FILENAME
        resolved_type = content_type or DEFAULT_TEXT_CONTENT_TYPE
        return await self.ingest_text(
            session,
            qdrant,
            filename=resolved_name,
            content=text,
            content_type=resolved_type,
        )
