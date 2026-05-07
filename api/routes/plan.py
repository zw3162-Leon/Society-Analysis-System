"""POST /plan — lightweight planner-only endpoint for evaluation."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["plan"])

_rewriter: Optional[object] = None
_planner: Optional[object] = None


def _get_rewriter():
    global _rewriter
    if _rewriter is None:
        from agents.query_rewriter import QueryRewriter
        from services.embeddings_service import EmbeddingsService
        from services.planner_memory import PlannerMemory
        mem = PlannerMemory()
        emb = EmbeddingsService()
        _rewriter = QueryRewriter(planner_memory=mem, embeddings=emb)
    return _rewriter


def _get_planner():
    global _planner
    if _planner is None:
        from agents.planner_v2 import BoundedPlannerV2
        from services.embeddings_service import EmbeddingsService
        from services.planner_memory import PlannerMemory
        mem = PlannerMemory()
        emb = EmbeddingsService()
        _planner = BoundedPlannerV2(planner_memory=mem, embeddings=emb)
    return _planner


class PlanRequest(BaseModel):
    question: str
    session_id: str = ""


class PlanResponse(BaseModel):
    planned_branches: list[str]


@router.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest) -> PlanResponse:
    rq = _get_rewriter().rewrite(request.question)
    branches = _get_planner().route_only(rq)
    return PlanResponse(planned_branches=branches)
