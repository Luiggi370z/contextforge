"""HTTP routes for document ingestion."""

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.documents.dependencies import get_arq_pool, get_document_service, get_qdrant
from app.api.v1.documents.schemas import (
    DocumentListResponse,
    DocumentResponse,
    IngestionJobResponse,
    IngestTextRequest,
)
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


@router.post("/upload", response_model=IngestionJobResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
    document_service: DocumentService = Depends(get_document_service),
    arq_pool=Depends(get_arq_pool),
) -> IngestionJobResponse:
    """Enqueue async ingestion; returns a job to poll at GET /v1/documents/jobs/{id}."""
    content = await file.read()
    try:
        job = await document_service.enqueue_upload(
            session,
            arq_pool,
            filename=file.filename,
            raw_bytes=content,
            content_type=file.content_type,
        )
    except IngestionError as error:
        raise domain_error_to_app_exception(error) from error
    increment_ingests()
    return IngestionJobResponse.model_validate(job)


@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
async def get_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    document_service: DocumentService = Depends(get_document_service),
) -> IngestionJobResponse:
    """Return the current state of an ingestion job, or 404."""
    job = await document_service.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return IngestionJobResponse.model_validate(job)
