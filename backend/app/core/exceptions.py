"""Application and domain exceptions."""

from __future__ import annotations

import uuid

import structlog

from app.core.constants import ERROR_INTERNAL_SERVER


def resolve_correlation_id(explicit: str | None = None) -> str:
    """Return request correlation id from context or a new UUID.

    Example:
        >>> resolve_correlation_id("abc") == "abc"
        True
    """
    if explicit:
        return explicit
    context = structlog.contextvars.get_contextvars()
    request_id = context.get("request_id")
    if isinstance(request_id, str) and request_id:
        return request_id
    return str(uuid.uuid4())


class AppException(Exception):
    """HTTP-facing exception raised only from API route handlers."""

    def __init__(
        self,
        detail: str,
        status_code: int,
        correlation_id: str | None = None,
    ) -> None:
        self.detail = detail
        self.status_code = status_code
        self.correlation_id = resolve_correlation_id(correlation_id)
        super().__init__(detail)


class DomainError(Exception):
    """Base class for errors raised from services (mapped in routes)."""


class DocumentNotFoundError(DomainError):
    """Raised when a document id does not exist."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}")


class ThreadNotFoundError(DomainError):
    """Raised when a thread id does not exist."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        super().__init__(f"Thread not found: {thread_id}")


class IngestionError(DomainError):
    """Raised when document ingestion fails."""


def domain_error_to_app_exception(error: DomainError) -> AppException:
    """Map a domain error to an HTTP AppException for route handlers."""
    from app.core.constants import (
        ERROR_DOCUMENT_NOT_FOUND,
        ERROR_INVALID_UPLOAD,
        ERROR_THREAD_NOT_FOUND,
    )

    if isinstance(error, DocumentNotFoundError):
        return AppException(detail=ERROR_DOCUMENT_NOT_FOUND, status_code=404)
    if isinstance(error, ThreadNotFoundError):
        return AppException(detail=ERROR_THREAD_NOT_FOUND, status_code=404)
    if isinstance(error, IngestionError):
        return AppException(detail=ERROR_INVALID_UPLOAD, status_code=400)
    return AppException(detail=ERROR_INTERNAL_SERVER, status_code=500)
