"""Manual data import endpoints for the Streamlit demo UI.

These endpoints intentionally stay thin: they validate UI input, create an
in-process background job, then call the same pipeline runners used by the
scheduler. Long-running imports therefore do not block the browser request.
"""
from __future__ import annotations

import threading
import uuid
from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/admin/import", tags=["admin-import"])


ImportMode = Literal["append", "overwrite"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class RedditImportRequest(BaseModel):
    subreddits: list[str] = Field(default_factory=lambda: ["worldnews"])
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    mode: ImportMode = "append"
    confirm_overwrite: bool = False
    limit_per_subreddit: int = Field(default=100, ge=1, le=500)
    comment_limit: int = Field(default=100, ge=0, le=500)
    include_comments: bool = True


class OfficialImportRequest(BaseModel):
    sources: list[str] = Field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    mode: ImportMode = "append"
    confirm_overwrite: bool = False
    write_chroma: bool = True


class ImportJob(BaseModel):
    job_id: str
    kind: Literal["reddit", "official"]
    status: JobStatus = "queued"
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    request: dict[str, Any]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


_jobs: dict[str, ImportJob] = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store(job: ImportJob) -> None:
    with _lock:
        _jobs[job.job_id] = job


def _update(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _jobs[job_id]
        for key, value in fields.items():
            setattr(job, key, value)


def _date_range_to_days_back(start_date: Optional[date]) -> int:
    if start_date is None:
        return 1
    delta = datetime.now(timezone.utc).date() - start_date
    return max(1, delta.days + 1)


def _normalize_subreddits(subreddits: list[str]) -> list[str]:
    cleaned = []
    for raw in subreddits:
        value = raw.strip().lstrip("r/").lower()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned or ["worldnews"]


def _delete_reddit_rows(subreddits: list[str]) -> dict[str, int]:
    """Delete retained PostgreSQL rows for the selected Reddit subreddits."""
    from services.postgres_service import PostgresService

    pg = PostgresService()
    deleted = {
        "post_entities_v2": 0,
        "posts_v2": 0,
        "orphan_topics_v2": 0,
        "orphan_entities_v2": 0,
        "orphan_claims_v2": 0,
    }
    with pg.cursor() as cur:
        cur.execute(
            """
            DELETE FROM post_entities_v2
            WHERE post_id IN (
                SELECT post_id FROM posts_v2
                WHERE source = 'reddit'
                  AND LOWER(COALESCE(subreddit, '')) = ANY(%s)
            )
            """,
            (subreddits,),
        )
        deleted["post_entities_v2"] = int(cur.rowcount or 0)
        cur.execute(
            """
            DELETE FROM posts_v2
            WHERE source = 'reddit'
              AND LOWER(COALESCE(subreddit, '')) = ANY(%s)
            """,
            (subreddits,),
        )
        deleted["posts_v2"] = int(cur.rowcount or 0)
        cur.execute(
            """
            DELETE FROM topics_v2 t
            WHERE NOT EXISTS (
                SELECT 1 FROM posts_v2 p WHERE p.topic_id = t.topic_id
            )
            """
        )
        deleted["orphan_topics_v2"] = int(cur.rowcount or 0)
        cur.execute(
            """
            DELETE FROM entities_v2 e
            WHERE NOT EXISTS (
                SELECT 1 FROM post_entities_v2 pe
                WHERE pe.entity_id = e.entity_id
            )
            """
        )
        deleted["orphan_entities_v2"] = int(cur.rowcount or 0)
        cur.execute(
            """
            DELETE FROM claims_v2 c
            WHERE NOT EXISTS (
                SELECT 1 FROM post_claims_v2 pc WHERE pc.claim_id = c.claim_id
            )
            """
        )
        deleted["orphan_claims_v2"] = int(cur.rowcount or 0)
    return deleted


def _delete_official_sources(sources: list[str]) -> dict[str, int]:
    """Delete retained Chroma official chunks for selected source names."""
    from services.chroma_collections import ChromaCollections

    cols = ChromaCollections()
    deleted: dict[str, int] = {}
    for source in sources:
        before = cols.official.count(where={"source": source})
        cols.official.delete(where={"source": source})
        deleted[source] = before
    return deleted


def _run_reddit_job(job_id: str, body: RedditImportRequest) -> None:
    _update(job_id, status="running", started_at=_now())
    warnings: list[str] = []
    try:
        from scripts.scheduler import _run_precompute_v2

        subreddits = _normalize_subreddits(body.subreddits)
        cleanup: dict[str, int] = {}
        if body.mode == "overwrite":
            cleanup = _delete_reddit_rows(subreddits)
            warnings.append(
                "Overwrite clears retained PostgreSQL Reddit rows for the "
                "selected subreddits before running the pipeline. Existing "
                "Kuzu graph nodes are upserted by the new run and may retain "
                "orphan historical nodes until a full graph rebuild."
            )
        result = _run_precompute_v2({
            "subreddits": subreddits,
            "reddit_days_back": _date_range_to_days_back(body.start_date),
            "reddit_limit_per_sub": body.limit_per_subreddit,
            "reddit_include_comments": body.include_comments,
            "reddit_comment_limit": body.comment_limit,
        })
        _update(
            job_id,
            status="succeeded",
            finished_at=_now(),
            result={"cleanup": cleanup, "pipeline": result},
            warnings=warnings,
        )
    except Exception as exc:
        _update(
            job_id,
            status="failed",
            finished_at=_now(),
            error=str(exc)[:1000],
            warnings=warnings,
        )


def _run_official_job(job_id: str, body: OfficialImportRequest) -> None:
    _update(job_id, status="running", started_at=_now())
    warnings: list[str] = []
    try:
        from agents.official_ingestion_pipeline import OfficialIngestionPipeline

        pipeline = OfficialIngestionPipeline(write_chroma=body.write_chroma)
        available = set(pipeline.list_sources())
        sources = [s.strip().lower() for s in body.sources if s.strip()]
        sources = sources or sorted(available)
        unknown = sorted(set(sources) - available)
        if unknown:
            raise ValueError(f"Unknown official source(s): {', '.join(unknown)}")

        cleanup: dict[str, int] = {}
        if body.mode == "overwrite" and body.write_chroma:
            cleanup = _delete_official_sources(sources)
        results: dict[str, int] = {}
        for source in sources:
            results[source] = pipeline.run_once(source_filter=source).get(source, 0)
        if body.start_date or body.end_date:
            warnings.append(
                "Official RSS import uses the configured feeds' current items. "
                "The date range is recorded by the job and used by the UI query "
                "context, but the RSS crawler does not currently backfill an "
                "arbitrary historical window."
            )
        _update(
            job_id,
            status="succeeded",
            finished_at=_now(),
            result={"cleanup": cleanup, "chunks": results},
            warnings=warnings,
        )
    except Exception as exc:
        _update(
            job_id,
            status="failed",
            finished_at=_now(),
            error=str(exc)[:1000],
            warnings=warnings,
        )


@router.post("/reddit", response_model=ImportJob)
def import_reddit(
    body: RedditImportRequest,
    background_tasks: BackgroundTasks,
) -> ImportJob:
    if body.mode == "overwrite" and not body.confirm_overwrite:
        raise HTTPException(
            status_code=400,
            detail="confirm_overwrite must be true for overwrite imports",
        )
    subreddits = _normalize_subreddits(body.subreddits)
    if not subreddits:
        raise HTTPException(status_code=400, detail="at least one subreddit is required")
    body.subreddits = subreddits
    job = ImportJob(
        job_id=f"import_{uuid.uuid4().hex[:12]}",
        kind="reddit",
        created_at=_now(),
        request=body.model_dump(mode="json"),
    )
    _store(job)
    background_tasks.add_task(_run_reddit_job, job.job_id, body)
    return job


@router.post("/official", response_model=ImportJob)
def import_official(
    body: OfficialImportRequest,
    background_tasks: BackgroundTasks,
) -> ImportJob:
    if body.mode == "overwrite" and not body.confirm_overwrite:
        raise HTTPException(
            status_code=400,
            detail="confirm_overwrite must be true for overwrite imports",
        )
    job = ImportJob(
        job_id=f"import_{uuid.uuid4().hex[:12]}",
        kind="official",
        created_at=_now(),
        request=body.model_dump(mode="json"),
    )
    _store(job)
    background_tasks.add_task(_run_official_job, job.job_id, body)
    return job


@router.get("/jobs/{job_id}", response_model=ImportJob)
def get_import_job(job_id: str) -> ImportJob:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
