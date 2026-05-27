from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from app.db.session import get_db
from app.ingestion.service import ingest_document_text
from app.retrieval.qdrant_store import get_qdrant_store
from app.schemas.documents import DocumentListResponse, DocumentResponse, IngestTextRequest

router = APIRouter()


@router.get("", response_model=DocumentListResponse)
async def list_documents(db: AsyncSession = Depends(get_db)) -> DocumentListResponse:
    total = await db.scalar(select(func.count()).select_from(Document)) or 0
    result = await db.execute(select(Document).order_by(Document.created_at.desc()).limit(100))
    items = [DocumentResponse.model_validate(d) for d in result.scalars().all()]
    return DocumentListResponse(items=items, total=total)


@router.post("", response_model=DocumentResponse, status_code=201)
async def ingest_text(
    body: IngestTextRequest, db: AsyncSession = Depends(get_db)
) -> DocumentResponse:
    qdrant = get_qdrant_store()
    doc = await ingest_document_text(
        db,
        qdrant,
        filename=body.filename,
        content=body.content,
        content_type=body.content_type,
    )
    return DocumentResponse.model_validate(doc)


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    qdrant = get_qdrant_store()
    doc = await ingest_document_text(
        db,
        qdrant,
        filename=file.filename or "upload.txt",
        content=text,
        content_type=file.content_type or "text/plain",
    )
    return DocumentResponse.model_validate(doc)
