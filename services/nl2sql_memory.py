"""
NL2SQL memory service - redesign-2026-05 Phase 2 + Phase 3.

Read/write wrapper around Chroma 2 (`chroma_nl2sql`). The collection holds
three kinds of documents identified by `metadata.kind`:

    kind=schema     - per-column descriptions written by Schema-aware Agent
    kind=success    - successful (NL, SQL) exemplars
    kind=error      - error patterns to avoid (auto-curated by Reflection)
    kind=guide      - durable NL2SQL rules / database notes / operator guidance

Conflict-replacement policy (Q11=B; PROJECT_REDESIGN_V2.md 7c-H):
    similarity < 0.92                 -> append (no conflict check)
    0.92 <= similarity < 0.95         -> direct overwrite (no LLM)
    similarity >= 0.95                -> LLM-arbitrated pairwise comparison

The service stays storage-only: it does not embed text itself. Callers must
supply embeddings (typically via `services.embeddings_service`).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import structlog

from config import (
    NL2SQL_CONFLICT_SIM_HIGH,
    NL2SQL_CONFLICT_SIM_LOW,
)
from services.chroma_collections import ChromaCollections

log = structlog.get_logger(__name__)


Kind = Literal["schema", "success", "error", "guide"]


# ── Document shape ───────────────────────────────────────────────────────────

@dataclass
class NL2SQLRecord:
    record_id: str
    kind: Kind
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


# ── Service ──────────────────────────────────────────────────────────────────

class NL2SQLMemory:
    """Per-`kind` upsert + retrieval over Chroma 2."""

    def __init__(
        self,
        collections: Optional[ChromaCollections] = None,
        sim_low: float = NL2SQL_CONFLICT_SIM_LOW,
        sim_high: float = NL2SQL_CONFLICT_SIM_HIGH,
        llm_judge: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self._cols = collections or ChromaCollections()
        self._sim_low = sim_low
        self._sim_high = sim_high
        # llm_judge(new_text, existing_text) -> True if new conflicts and
        # should replace existing. Defaults to a permissive judge.
        self._llm_judge = llm_judge or (lambda new_text, old_text: True)

    # ── Upsert paths ───────────────────────────────────────────────────────────

    def upsert_schema(
        self,
        table_name: str,
        column_name: str,
        text: str,
        embedding: list[float],
        fingerprint: str,
        sample_values: Optional[list[str]] = None,
    ) -> str:
        """Schema docs use a deterministic id so that re-runs overwrite cleanly."""
        record_id = f"schema::{table_name}::{column_name}"
        meta = {
            "kind": "schema",
            "table_name": table_name,
            "column_name": column_name,
            "fingerprint": fingerprint,
            "sample_values": json.dumps(sample_values or [])[:1500],
            "updated_at": time.time(),
        }
        self._cols.nl2sql.upsert(
            ids=[record_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )
        return record_id

    def upsert_success(
        self,
        nl_query: str,
        sql_query: str,
        embedding: list[float],
        table_hints: Optional[list[str]] = None,
    ) -> str:
        """Add a successful (NL, SQL) exemplar with conflict-aware replacement."""
        text = f"NL: {nl_query}\nSQL: {sql_query}"
        meta = {
            "kind": "success",
            "table_hints": json.dumps(table_hints or []),
            "confidence": 0.5,
            "hit_count": 0,
            "last_used_at": time.time(),
        }
        return self._upsert_with_conflict_check(text, embedding, meta)

    def upsert_error(
        self,
        failure_reason: str,
        bad_pattern: str,
        embedding: list[float],
        table_hints: Optional[list[str]] = None,
    ) -> str:
        """Record an error pattern (NL2SQL repair loop or Critic-detected)."""
        text = f"Avoid: {bad_pattern}\nReason: {failure_reason}"
        meta = {
            "kind": "error",
            "table_hints": json.dumps(table_hints or []),
            "confidence": 0.5,
            "hit_count": 0,
            "last_used_at": time.time(),
        }
        return self._upsert_with_conflict_check(text, embedding, meta)

    def upsert_guidance(
        self,
        rule_id: str,
        text: str,
        embedding: list[float],
        category: str = "rule",
        priority: int = 50,
    ) -> str:
        """Store durable NL2SQL guidance in Chroma 2 with a stable id."""
        safe_rule_id = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_"
            for ch in rule_id.strip().lower()
        ) or "unnamed"
        record_id = f"guide::{safe_rule_id}"
        meta = {
            "kind": "guide",
            "category": category,
            "priority": int(priority),
            "updated_at": time.time(),
        }
        self._cols.nl2sql.upsert(
            ids=[record_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )
        return record_id

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def recall_schema(
        self, embedding: list[float], n_results: int = 8,
        table_filter: Optional[str] = None,
    ) -> list[dict]:
        where: dict = {"kind": "schema"}
        if table_filter:
            where["table_name"] = table_filter
        return self._cols.nl2sql.query(
            embedding=embedding, n_results=n_results, where=where,
        )

    def recall_success(self, embedding: list[float],
                       n_results: int = 5) -> list[dict]:
        return self._recall_with_hit(embedding, "success", n_results)

    def recall_errors(self, embedding: list[float],
                      n_results: int = 3) -> list[dict]:
        return self._recall_with_hit(embedding, "error", n_results)

    def recall_guidance(self, embedding: list[float],
                        n_results: int = 8) -> list[dict]:
        return self._recall_with_hit(embedding, "guide", n_results)

    def count_guidance(self) -> int:
        return self._cols.nl2sql.count(where={"kind": "guide"})

    def prune_stale_schema(
        self,
        live_columns: set[tuple[str, str]],
    ) -> list[str]:
        """Delete Chroma 2 schema records for columns absent from Postgres."""
        if not live_columns:
            return []
        try:
            results = self._cols.nl2sql.handle.get(
                where={"kind": "schema"},
                include=["metadatas"],
            )
        except Exception as exc:
            log.warning("nl2sql_memory.prune_schema_get_failed",
                        error=str(exc)[:160])
            return []

        stale_ids: list[str] = []
        ids = results.get("ids") or []
        metas = results.get("metadatas") or []
        for idx, record_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            table = str((meta or {}).get("table_name") or "").strip()
            column = str((meta or {}).get("column_name") or "").strip()
            if table and column and (table, column) not in live_columns:
                stale_ids.append(str(record_id))

        if stale_ids:
            self._cols.nl2sql.delete(ids=stale_ids)
            log.info("nl2sql_memory.pruned_stale_schema",
                     count=len(stale_ids),
                     ids=stale_ids[:20])
        return stale_ids

    # ── Auto-curation hooks ────────────────────────────────────────────────────

    def delete_records(self, record_ids: list[str]) -> None:
        """Used by Reflection's ablation step to drop a causal record."""
        if record_ids:
            self._cols.nl2sql.delete(ids=record_ids)
            log.info("nl2sql_memory.deleted", count=len(record_ids))

    def clear_all_errors(self) -> int:
        """Delete every error-kind record.

        Use this to reset accumulated bad lessons that are blocking valid
        SQL generation (e.g. empty-SQL lessons that discourage the LLM from
        writing queries for certain question types).
        """
        try:
            results = self._cols.nl2sql.handle.get(
                where={"kind": "error"},
                include=[],
            )
            ids = results.get("ids") or []
            if ids:
                self._cols.nl2sql.delete(ids=ids)
                log.info("nl2sql_memory.cleared_all_errors", count=len(ids))
            return len(ids)
        except Exception as exc:
            log.warning("nl2sql_memory.clear_errors_failed",
                        error=str(exc)[:160])
            return 0

    # ── Internals ──────────────────────────────────────────────────────────────

    def _upsert_with_conflict_check(
        self, text: str, embedding: list[float], metadata: dict,
    ) -> str:
        """3-tier conflict policy from PROJECT_REDESIGN_V2.md 7c-H."""
        kind = metadata.get("kind", "")
        existing = self._cols.nl2sql.query(
            embedding=embedding, n_results=5, where={"kind": kind},
        )

        # Find the most similar existing record (highest similarity)
        if existing:
            top = existing[0]
            sim = float(top.get("similarity", 0.0))
            old_id = top["id"]
            old_text = top.get("document", "")
            if sim >= self._sim_high:
                if self._llm_judge(text, old_text):
                    log.info("nl2sql_memory.replace.llm_arbitrated",
                             old_id=old_id, similarity=round(sim, 3))
                    self._cols.nl2sql.delete(ids=[old_id])
                else:
                    log.info("nl2sql_memory.coexist.llm_says_not_conflict",
                             similarity=round(sim, 3))
            elif sim >= self._sim_low:
                log.info("nl2sql_memory.replace.direct",
                         old_id=old_id, similarity=round(sim, 3))
                self._cols.nl2sql.delete(ids=[old_id])
            # else: sim < sim_low -> append (no replacement)

        record_id = f"{kind}::{uuid.uuid4().hex}"
        self._cols.nl2sql.upsert(
            ids=[record_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return record_id

    def _recall_with_hit(
        self, embedding: list[float], kind: Kind, n_results: int,
    ) -> list[dict]:
        results = self._cols.nl2sql.query(
            embedding=embedding, n_results=n_results, where={"kind": kind},
        )
        # Bump hit counters / last_used_at (best-effort; failures are non-fatal)
        for r in results:
            try:
                meta = dict(r.get("metadata") or {})
                meta["hit_count"] = int(meta.get("hit_count", 0)) + 1
                meta["last_used_at"] = time.time()
                self._cols.nl2sql.handle.update(  # type: ignore[attr-defined]
                    ids=[r["id"]],
                    metadatas=[meta],
                )
            except Exception as exc:
                log.warning("nl2sql_memory.hit_update_error",
                            id=r.get("id"), error=str(exc)[:120])
        return results
