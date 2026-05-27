import pytest
from fastapi import APIRouter, FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.constants import ERROR_INTERNAL_SERVER
from app.core.exception_handlers import register_exception_handlers
from app.core.exceptions import AppException


def _build_test_app() -> FastAPI:
    application = FastAPI(debug=False)
    register_exception_handlers(application)
    return application


@pytest.mark.asyncio
async def test_app_exception_returns_json_error_body():
    application = _build_test_app()
    probe_router = APIRouter()

    @probe_router.get("/probe-app-error")
    async def probe_app_error() -> None:
        raise AppException(detail="Bad request sample", status_code=400)

    application.include_router(probe_router)
    async with AsyncClient(
        transport=ASGITransport(app=application, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/probe-app-error")

    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Bad request sample"
    assert "correlation_id" in body


@pytest.mark.asyncio
async def test_unhandled_exception_returns_500_without_leaking_details():
    application = _build_test_app()
    probe_router = APIRouter()

    @probe_router.get("/probe-unhandled")
    async def probe_unhandled() -> None:
        raise RuntimeError("secret internal detail")

    application.include_router(probe_router)
    async with AsyncClient(
        transport=ASGITransport(app=application, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.get("/probe-unhandled")

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == ERROR_INTERNAL_SERVER
    assert "secret internal detail" not in body["detail"]
