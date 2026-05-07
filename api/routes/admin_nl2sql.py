"""Admin endpoints for NL2SQL memory management."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/admin/nl2sql", tags=["admin-nl2sql"])


@router.post("/clear-errors")
def clear_nl2sql_errors() -> dict:
    """Delete all accumulated NL2SQL error lessons from Chroma 2.

    Call this when error lessons have become stale or counterproductive
    (e.g. 'no SQL produced' lessons that block valid query generation).
    The builtin guidance rules are not affected.
    """
    from services.nl2sql_memory import NL2SQLMemory
    mem = NL2SQLMemory()
    deleted = mem.clear_all_errors()
    return {"deleted_error_lessons": deleted}


@router.get("/error-count")
def count_nl2sql_errors() -> dict:
    """Return the number of error-kind records currently in Chroma 2."""
    from services.chroma_collections import ChromaCollections
    cols = ChromaCollections()
    count = cols.nl2sql.count(where={"kind": "error"})
    return {"error_lesson_count": count}
