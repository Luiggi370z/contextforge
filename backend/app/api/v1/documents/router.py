"""HTTP routes for document ingestion."""

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.documents.dependencies import get_document_service, get_qdrant
from app.api.v1.documents.schemas import DocumentListResponse, DocumentResponse, IngestTextRequest
from app.api.v1.documents.service import DocumentService
from app.api.v1.metrics.service import increment_ingests
from app.core.exceptions import IngestionError, domain_error_to_app_exception
from app.db.session import get_db
from app.retrieval.qdrant_store import QdrantStore

router = APIRouter()


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    session: AsyncSession = Depends(get_db),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List ingested documents."""
    result = await document_service.list_documents(session)
    items = [DocumentResponse.model_validate(doc) for doc in result.items]
    return DocumentListResponse(items=items, total=result.total)


@router.post("", response_model=DocumentResponse, status_code=201)
async def ingest_text(
    body: IngestTextRequest,
    session: AsyncSession = Depends(get_db),
    document_service: DocumentService = Depends(get_document_service),
    qdrant: QdrantStore = Depends(get_qdrant),
) -> DocumentResponse:
    """Ingest a document from JSON body."""
    try:
        document = await document_service.ingest_text(
            session,
            qdrant,
            filename=body.filename,
            content=body.content,
            content_type=body.content_type,
        )
    except IngestionError as error:
        raise domain_error_to_app_exception(error) from error
    increment_ingests()
    return DocumentResponse.model_validate(document)


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
    document_service: DocumentService = Depends(get_document_service),
    qdrant: QdrantStore = Depends(get_qdrant),
) -> DocumentResponse:
    """Ingest a document from multipart upload."""
    content = await file.read()
    try:
        document = await document_service.ingest_upload(
            session,
            qdrant,
            filename=file.filename,
            raw_bytes=content,
            content_type=file.content_type,
        )
    except IngestionError as error:
        raise domain_error_to_app_exception(error) from error
    increment_ingests()
    return DocumentResponse.model_validate(document)
