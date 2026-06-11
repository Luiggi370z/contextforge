"""Eval run + history routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.eval.schemas import EvalRunListResponse, EvalRunRequest, EvalRunResponse
from app.api.v1.eval.service import EvalService
from app.db.session import get_db

router = APIRouter()


def get_eval_service() -> EvalService:
    return EvalService()


@router.post("/run", response_model=EvalRunListResponse, status_code=201)
async def run_eval(
    body: EvalRunRequest,
    session: AsyncSession = Depends(get_db),
    service: EvalService = Depends(get_eval_service),
) -> EvalRunListResponse:
    runs = await service.run(session, configs=body.configs)
    items = [EvalRunResponse.model_validate(run) for run in runs]
    return EvalRunListResponse(items=items, total=len(items))


@router.get("/runs", response_model=EvalRunListResponse)
async def list_runs(
    session: AsyncSession = Depends(get_db),
    service: EvalService = Depends(get_eval_service),
) -> EvalRunListResponse:
    items, total = await service.list_runs(session)
    return EvalRunListResponse(
        items=[EvalRunResponse.model_validate(run) for run in items], total=total
    )
