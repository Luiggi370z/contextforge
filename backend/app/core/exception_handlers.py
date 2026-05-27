"""Register global FastAPI exception handlers."""

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.constants import ERROR_INTERNAL_SERVER
from app.core.exceptions import AppException
from app.schemas.errors import ErrorResponse

log = structlog.get_logger(__name__)


def register_exception_handlers(application: FastAPI) -> None:
    """Attach AppException and fallback handlers to the FastAPI app."""

    @application.exception_handler(AppException)
    async def app_exception_handler(
        request: Request, exc: AppException
    ) -> JSONResponse:
        log.warning(
            "handled_app_exception",
            status_code=exc.status_code,
            detail=exc.detail,
            correlation_id=exc.correlation_id,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                detail=exc.detail,
                correlation_id=exc.correlation_id,
            ).model_dump(mode="json", by_alias=True),
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log.error(
            "unhandled_exception",
            error=str(exc),
            path=request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(detail=ERROR_INTERNAL_SERVER).model_dump(
                mode="json", by_alias=True
            ),
        )
