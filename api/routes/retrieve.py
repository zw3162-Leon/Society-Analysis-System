"""
/retrieve/{evidence,nl2sql,kg} - redesign-2026-05 Phase 3.7.

Lets each Phase 3 branch be invoked independently, both for debug /
evaluation and for external integrations that don't want to go through
Query Rewrite + Planner.

The handlers stay thin: they construct the branch tool, delegate, and
return the Pydantic output untouched. Lazy construction keeps cold-start
cheap when only one branch is exercised.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


# ── A. Evidence ──────────────────────────────────────────────────────────────

class EvidenceRequest(BaseModel):
    query: str
    metadata_filter: Optional[dict] = None
    rerank: bool = True


@router.post("/evidence")
def retrieve_evidence(req: EvidenceRequest) -> dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    try:
        from tools.hybrid_retrieval import HybridRetriever
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"branch unavailable: {exc}")
    bundle = HybridRetriever().retrieve(
        req.query,
        metadata_filter=req.metadata_filter,
        rerank=req.rerank,
    )
    return {"branch": "evidence", "bundle": bundle.model_dump()}


# ── B. NL2SQL ────────────────────────────────────────────────────────────────

class NL2SQLRequest(BaseModel):
    nl_query: str = Field(..., description="natural-language question")


@router.post("/nl2sql")
def retrieve_nl2sql(req: NL2SQLRequest) -> dict[str, Any]:
    if not req.nl_query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    try:
        from tools.nl2sql_tools import NL2SQLTool
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"branch unavailable: {exc}")
    out = NL2SQLTool().answer(req.nl_query)
    return {"branch": "nl2sql", "sql_output": out.model_dump()}


# ── C. KG Query ──────────────────────────────────────────────────────────────

class KGRequest(BaseModel):
    query_kind: str  # propagation_path | key_nodes | topic_correlation | cascade_tree |
                     # viral_cascade | echo_chamber | influencer_rank | coordinated_groups
    target: dict[str, Any] = Field(default_factory=dict)


@router.post("/kg")
def retrieve_kg(req: KGRequest) -> dict[str, Any]:
    # Analytics query kinds use KGAnalytics (NetworkX-backed algorithms)
    if req.query_kind in ("echo_chamber", "influencer_rank", "coordinated_groups"):
        try:
            from agents.kg_analytics import KGAnalytics
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"analytics unavailable: {exc}")
        analytics = KGAnalytics()
        if req.query_kind == "echo_chamber":
            out = analytics.echo_chamber(
                topic_id=req.target.get("topic_id", ""),
                modularity_threshold=float(req.target.get("modularity_threshold", 0.3)),
            )
        elif req.query_kind == "influencer_rank":
            out = analytics.influencer_rank(
                topic_id=req.target.get("topic_id"),
                top_k=int(req.target.get("top_k", 10)),
                since_days=req.target.get("since_days", 30),
            )
        else:  # coordinated_groups
            out = analytics.coordinated_groups(
                topic_id=req.target.get("topic_id"),
                min_size=int(req.target.get("min_size", 3)),
                since_days=req.target.get("since_days", 30),
            )
        return {"branch": "kg", "kg_output": out.model_dump()}

    try:
        from tools.kg_query_tools import KGQueryTool
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"branch unavailable: {exc}")
    tool = KGQueryTool()
    if req.query_kind == "propagation_path":
        out = tool.propagation_path(
            source_account=req.target.get("source_account", ""),
            target_account=req.target.get("target_account", ""),
            max_hops=int(req.target.get("max_hops", 4)),
        )
    elif req.query_kind == "key_nodes":
        out = tool.key_nodes(
            topic_id=req.target.get("topic_id", ""),
            top_k=int(req.target.get("top_k", 10)),
        )
    elif req.query_kind == "topic_correlation":
        out = tool.topic_correlation(
            topic_a=req.target.get("topic_a", ""),
            topic_b=req.target.get("topic_b", ""),
        )
    elif req.query_kind == "cascade_tree":
        out = tool.cascade_tree(
            root_post_id=req.target.get("root_post_id", ""),
            max_depth=int(req.target.get("max_depth", 10)),
        )
    elif req.query_kind == "viral_cascade":
        out = tool.viral_cascade(
            topic_id=req.target.get("topic_id", ""),
            top_k=int(req.target.get("top_k", 5)),
        )
    else:
        raise HTTPException(status_code=400,
                            detail=f"unknown query_kind: {req.query_kind}")
    return {"branch": "kg", "kg_output": out.model_dump()}
