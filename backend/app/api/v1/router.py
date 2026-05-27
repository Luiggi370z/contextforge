from fastapi import APIRouter

from app.api.v1 import documents, health, query, threads

api_router = APIRouter(prefix="/v1")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(threads.router, prefix="/threads", tags=["threads"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
