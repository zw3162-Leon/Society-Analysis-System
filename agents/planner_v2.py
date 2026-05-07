"""
Bounded Planner v2 - redesign-2026-05 Phase 4.2.

Replaces the capability-template-based v1 planner. Design (PROJECT_REDESIGN_V2.md
1.2 + 5c):

    rewritten_query (Subtasks)
       -> Planner picks branches per subtask (Router-style decision)
       -> Workflow: which branches in parallel, which sequenced
       -> Execute each branch in parallel within the bounded DAG
       -> PlanExecutionV2 (per-branch outputs + errors + which exemplars used)

Bounds:
    - <= 3 branches running in parallel per subtask
    - <= 5 branch executions per workflow total (across all subtasks)

Q9 confidence rule (PROJECT_REDESIGN_V2.md 7c-B):
    - count_branch_combo_successes(branches) >= 3  -> high confidence;
                                                       skip Chroma 3 few-shot
    - else                                          -> recall few-shot

Few-shot is currently advisory (logged + carried in PlanExecutionV2.notes);
Phase 4.3 Report Writer doesn't need it directly because the Subtask carries
the suggested branches already. We keep the recall hook so Phase 5 can
back-propagate signal into Chroma 3.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import structlog
from pydantic import BaseModel, Field

from config import KG_TOPIC_SEMANTIC_FALLBACK_MIN_SIM
from models.branch_output import (
    BranchExecutionStatus,
    EvidenceOutput,
    KGNode,
    KGOutput,
    SQLAttempt,
    SQLOutput,
)
from models.query import RewrittenQuery, Subtask

log = structlog.get_logger(__name__)


BranchName = str  # "evidence" | "nl2sql" | "kg"


# ── Data contracts ──────────────────────────────────────────────────────────

class BranchInvocation(BaseModel):
    """One concrete branch call this Planner intends to run."""

    subtask_index: int
    branch: BranchName
    payload: dict = Field(default_factory=dict)


class BranchResult(BaseModel):
    invocation: BranchInvocation
    status: BranchExecutionStatus
    output: Optional[dict] = None  # branch_output dict (model_dump)


class PlanExecutionV2(BaseModel):
    workflow: list[BranchInvocation] = Field(default_factory=list)
    results: list[BranchResult] = Field(default_factory=list)
    branches_used: list[BranchName] = Field(default_factory=list)
    used_few_shot: bool = False
    notes: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0


@dataclass
class _BranchRouter:
    """Map subtask intent -> default branch set (Router responsibility).

    Multi-branch by default: most intents benefit from cross-source
    triangulation (raw counts from NL2SQL, who is amplifying it from KG,
    what authoritative sources said from Evidence). The Planner caps
    parallelism via `max_parallel_branches` (default 3) so even fan-out
    here stays bounded.
    """

    intent_branch_map: dict[str, list[BranchName]] = field(default_factory=lambda: {
        # Pure SQL aggregation - one branch is enough.
        "community_count":    ["nl2sql"],
        # Listing posts is helped by who-posted-with-whom context.
        "community_listing":  ["nl2sql", "kg"],
        # Trends are fundamentally multi-signal: counts + key spreaders.
        "trend":              ["nl2sql", "kg"],
        # Propagation always wants supporting volume from SQL alongside
        # the structural picture.
        "propagation":        ["kg", "nl2sql"],
        # Fact-check NEEDS official sources, but the community angle
        # (who is actually saying it on Reddit) is part of the answer.
        "fact_check":         ["evidence", "nl2sql"],
        "topic_claim_audit":  ["nl2sql", "evidence"],
        # Recap an authoritative source. Add nl2sql so we can say
        # whether the community discussed it.
        "official_recap":     ["evidence", "nl2sql"],
        # Comparison is the canonical 3-source case.
        "comparison":         ["evidence", "nl2sql", "kg"],
        # Explaining a decision needs structural + factual.
        "explain_decision":   ["nl2sql", "kg"],
        # Freeform: cast a wide net.
        "freeform":           ["evidence", "nl2sql"],
        # ── KG-specialised (Phase C) ─────────────────────────────────────
        "propagation_trace":   ["kg"],
        "influencer_query":    ["kg", "nl2sql"],
        "coordination_check":  ["kg"],
        "community_structure": ["kg", "nl2sql"],
        "cascade_query":       ["kg"],
    })

    def route(self, subtask: Subtask) -> list[BranchName]:
        # Suggested branches from Query Rewriter take precedence
        if subtask.suggested_branches:
            return list(subtask.suggested_branches)
        return list(self.intent_branch_map.get(subtask.intent, ["evidence", "nl2sql"]))


# ── Planner ─────────────────────────────────────────────────────────────────

_TOPIC_SCOPED_INTENTS = {
    "community_count", "community_listing", "trend",
    "propagation", "explain_decision", "topic_claim_audit",
    # KG-specialised intents that benefit from semantic topic resolution
    "propagation_trace", "influencer_query", "coordination_check",
    "community_structure", "cascade_query",
}


# KG-specialised intent → (KG analytics method, default kwargs).
# `propagation_trace` is handled separately (it expects two account ids).
_KG_INTENT_TO_METHOD = {
    "influencer_query":    ("influencer_rank",    {}),
    "coordination_check":  ("coordinated_groups", {}),
    "community_structure": ("echo_chamber",       {}),
    "cascade_query":       ("viral_cascade",      {}),  # via KGQueryTool
    "propagation_trace":   ("propagation_path",   {}),  # via KGQueryTool
}


@dataclass
class BoundedPlannerV2:
    """Plans + executes branch calls for one rewritten query."""

    evidence_runner: Optional[Callable[[Subtask], EvidenceOutput]] = None
    nl2sql_runner: Optional[Callable[[Subtask], SQLOutput]] = None
    kg_runner: Optional[Callable[[Subtask], KGOutput]] = None

    planner_memory: Optional[Any] = None  # services.planner_memory.PlannerMemory
    embeddings: Optional[Any] = None      # services.embeddings_service.EmbeddingsService
    topic_resolver: Optional[Any] = None  # tools.topic_resolver.TopicResolver

    max_parallel_branches: int = 3
    max_branch_calls: int = 5
    confidence_hit_count: int = 3

    # ── Public ───────────────────────────────────────────────────────────────

    def plan_and_execute(self, rq: RewrittenQuery) -> PlanExecutionV2:
        t0 = time.monotonic()
        plan = self._plan(rq)
        execution = self._execute(plan, rq)
        execution.elapsed_ms = int((time.monotonic() - t0) * 1000)

        unique_branches: list[BranchName] = []
        for inv in execution.workflow:
            if inv.branch not in unique_branches:
                unique_branches.append(inv.branch)
        execution.branches_used = unique_branches

        log.info("planner_v2.done",
                 subtasks=len(rq.subtasks),
                 invocations=len(execution.workflow),
                 ok=sum(1 for r in execution.results if r.status.success),
                 fail=sum(1 for r in execution.results if not r.status.success),
                 used_few_shot=execution.used_few_shot,
                 elapsed_ms=execution.elapsed_ms)
        return execution

    # ── Planning ─────────────────────────────────────────────────────────────

    def _plan(self, rq: RewrittenQuery) -> list[BranchInvocation]:
        """Translate subtasks into branch invocations."""
        router = _BranchRouter()
        invocations: list[BranchInvocation] = []
        for idx, sub in enumerate(rq.subtasks):
            branches = router.route(sub)[: self.max_parallel_branches]
            if _is_broad_topic_listing(sub):
                branches = ["nl2sql"]
            # Pre-resolve semantic topic match for topic-scoped subtasks
            # so NL2SQL / KG get a concrete topic_id list to filter on.
            resolved_topic_ids = self._resolve_topics_for_subtask(sub)
            for branch in branches:
                if len(invocations) >= self.max_branch_calls:
                    break
                invocations.append(BranchInvocation(
                    subtask_index=idx,
                    branch=branch,
                    payload=self._build_payload(
                        branch, sub, resolved_topic_ids,
                    ),
                ))
            if len(invocations) >= self.max_branch_calls:
                break
        return invocations

    def _resolve_topics_for_subtask(self, sub: Subtask) -> list[dict]:
        """Run the semantic topic resolver. Returns list of {topic_id, label, similarity}."""
        if _is_broad_topic_listing(sub):
            return []
        # If the rewriter already pinned a real topic_id, trust it. Do not
        # trust natural-language labels here: the rewriter can inherit labels
        # from prior model prose (for example "Global Politics"), and passing
        # those into Kuzu as ids yields empty graph results.
        if sub.targets.topic_id and _looks_like_topic_id(sub.targets.topic_id):
            return [{
                "topic_id": sub.targets.topic_id,
                "label": "",
                "similarity": 1.0,
            }]
        # Only run for topic-scoped intents, unless we have a non-id topic
        # target that needs coercion.
        if sub.intent not in _TOPIC_SCOPED_INTENTS and not sub.targets.topic_id:
            return []
        if self.topic_resolver is None:
            try:
                from tools.topic_resolver import TopicResolver
                self.topic_resolver = TopicResolver()
            except Exception as exc:
                log.warning("planner_v2.topic_resolver_unavailable",
                            error=str(exc)[:160])
                return []
        try:
            phrase = sub.targets.topic_id or sub.text
            include_alternatives = _subtask_needs_kg_signal(sub)
            if hasattr(self.topic_resolver, "resolve_candidates"):
                matches = self.topic_resolver.resolve_candidates(
                    phrase,
                    top_k=6 if include_alternatives else 3,
                    include_semantic_alternatives=include_alternatives,
                    min_similarity=(
                        KG_TOPIC_SEMANTIC_FALLBACK_MIN_SIM
                        if include_alternatives else None
                    ),
                )
            else:
                matches = self.topic_resolver.resolve(phrase, top_k=3)
        except Exception as exc:
            log.warning("planner_v2.topic_resolve_error",
                        error=str(exc)[:160])
            return []
        if not matches:
            return []
        log.info("planner_v2.topic_resolved",
                 phrase=sub.text[:80],
                 matches=[(m.topic_id, m.label, round(m.similarity, 3))
                          for m in matches])
        return [
            {"topic_id": m.topic_id, "label": m.label,
             "similarity": round(m.similarity, 3)}
            for m in matches
        ]

    @staticmethod
    def _build_payload(
        branch: BranchName, sub: Subtask,
        resolved_topic_ids: Optional[list[dict]] = None,
    ) -> dict:
        resolved_topic_ids = resolved_topic_ids or []
        topic_id_list = [t["topic_id"] for t in resolved_topic_ids]

        if branch == "evidence":
            query = sub.text
            if sub.intent == "topic_claim_audit" and resolved_topic_ids:
                labels = ", ".join(
                    t.get("label") or t.get("topic_id", "")
                    for t in resolved_topic_ids
                )
                query = (
                    f"Official/evidence sources relevant to claims in topic "
                    f"{labels}: {sub.text}"
                )
            return {
                "query": query,
                "metadata_filter": sub.targets.metadata_filter,
            }
        if branch == "nl2sql":
            payload = {"nl_query": sub.text, "intent": sub.intent}
            if topic_id_list:
                # Append a structured hint that NL2SQL's prompt knows
                # to consume.
                payload["topic_id_hints"] = resolved_topic_ids
            return payload
        if branch == "kg":
            payload: dict[str, Any] = {"intent": sub.intent, "query": sub.text}
            # KG runner picks a single topic_id; use the top match.
            if topic_id_list:
                payload["topic_id"] = topic_id_list[0]
                payload["topic_id_hints"] = resolved_topic_ids
            elif sub.targets.topic_id:
                payload["topic_id"] = sub.targets.topic_id
            if sub.targets.account_id:
                payload["account_id"] = sub.targets.account_id
            # propagation_trace needs two account anchors; the rewriter
            # may stash them in metadata_filter or in the subtask text.
            if sub.intent == "propagation_trace":
                meta = sub.targets.metadata_filter or {}
                src = meta.get("source_account")
                dst = meta.get("target_account")
                # Fallback: scan the subtask text for `u_*` style account
                # ids (and a few common shapes like @handle / "user_x") so
                # we don't return "missing source/target" just because the
                # Rewriter forgot to populate metadata_filter.
                if not (src and dst):
                    candidates = _extract_account_ids(sub.text)
                    if not src and candidates:
                        src = candidates[0]
                    if not dst and len(candidates) >= 2:
                        dst = candidates[1]
                if src:
                    payload["source_account"] = src
                if dst:
                    payload["target_account"] = dst
            return payload
        return {}

    # ── Execution ────────────────────────────────────────────────────────────

    def _execute(
        self, plan: list[BranchInvocation], rq: RewrittenQuery,
    ) -> PlanExecutionV2:
        execution = PlanExecutionV2(workflow=list(plan))
        if not plan:
            execution.notes.append("empty_plan")
            return execution

        # Q9 confidence rule + few-shot recall (advisory)
        unique_branches = sorted({inv.branch for inv in plan})
        execution.used_few_shot = self._should_use_few_shot(
            rq, unique_branches,
        )

        # Parallel execution. Each branch is independent at the contract
        # level (they read different stores).
        with ThreadPoolExecutor(
            max_workers=self.max_parallel_branches,
        ) as pool:
            futures = {
                pool.submit(self._run_one, inv): inv
                for inv in plan
            }
            for fut in as_completed(futures):
                inv = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    log.error("planner_v2.branch_exception",
                              branch=inv.branch, error=str(exc)[:160])
                    result = BranchResult(
                        invocation=inv,
                        status=BranchExecutionStatus(
                            branch=inv.branch, success=False,
                            error=str(exc)[:200],
                            error_kind="branch_exception",
                        ),
                    )
                execution.results.append(result)

        # Restore stable order (by original plan index)
        order = {id(inv): i for i, inv in enumerate(plan)}
        execution.results.sort(key=lambda r: order.get(id(r.invocation), 0))
        return execution

    def _should_use_few_shot(
        self, rq: RewrittenQuery, unique_branches: list[BranchName],
    ) -> bool:
        if self.planner_memory is None or self.embeddings is None:
            return False
        try:
            self._maintain_planner_memory()
            hits = self.planner_memory.count_branch_combo_successes(unique_branches)
            if hits >= self.confidence_hit_count:
                return False
            # Low confidence: recall a few exemplars (advisory)
            embedding = self.embeddings.embed(rq.original)
            self.planner_memory.recall_workflow_exemplars(
                embedding, n_results=5,
            )
            return True
        except Exception as exc:
            log.warning("planner_v2.few_shot_recall_error",
                        error=str(exc)[:120])
            return False

    def _maintain_planner_memory(self) -> None:
        """Keep Chroma 3 topic references aligned with live Postgres topics."""
        if getattr(self, "_planner_memory_maintained", False):
            return
        try:
            from services.postgres_service import PostgresService
            pg = PostgresService()
            pg.connect()
            with pg.cursor() as cur:
                cur.execute("SELECT topic_id FROM topics_v2")
                live_topic_ids = {
                    str(row["topic_id"])
                    for row in (cur.fetchall() or [])
                    if row.get("topic_id")
                }
            if live_topic_ids and hasattr(
                self.planner_memory, "prune_stale_topic_references",
            ):
                self.planner_memory.prune_stale_topic_references(
                    live_topic_ids,
                )
        except Exception as exc:
            log.warning("planner_v2.planner_memory_maintenance_failed",
                        error=str(exc)[:160])
        finally:
            self._planner_memory_maintained = True

    def _run_one(self, inv: BranchInvocation) -> BranchResult:
        t0 = time.monotonic()
        runner = self._select_runner(inv.branch)
        if runner is None:
            return BranchResult(
                invocation=inv,
                status=BranchExecutionStatus(
                    branch=inv.branch, success=False,
                    error=f"no runner registered for branch {inv.branch}",
                    error_kind="branch_missing",
                ),
            )
        try:
            output_obj = runner(inv)
            elapsed = int((time.monotonic() - t0) * 1000)
            return BranchResult(
                invocation=inv,
                status=BranchExecutionStatus(
                    branch=inv.branch, success=True, elapsed_ms=elapsed,
                ),
                output=output_obj.model_dump() if hasattr(
                    output_obj, "model_dump") else output_obj,
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            log.error("planner_v2.runner_error",
                      branch=inv.branch, error=str(exc)[:160])
            return BranchResult(
                invocation=inv,
                status=BranchExecutionStatus(
                    branch=inv.branch, success=False,
                    error=str(exc)[:200], error_kind="branch_runner_error",
                    elapsed_ms=elapsed,
                ),
            )

    def _select_runner(self, branch: BranchName):
        if branch == "evidence":
            return self.evidence_runner
        if branch == "nl2sql":
            return self.nl2sql_runner
        if branch == "kg":
            return self.kg_runner
        return None


# ── Default branch runners ──────────────────────────────────────────────────
# These thin wrappers map (BranchInvocation) -> (EvidenceOutput | SQLOutput |
# KGOutput) using the Phase 3 tools. Tests / unusual deployments can pass in
# their own runners.

def default_evidence_runner(inv: BranchInvocation) -> EvidenceOutput:
    from tools.hybrid_retrieval import HybridRetriever
    bundle = HybridRetriever().retrieve(
        query=inv.payload.get("query", ""),
        metadata_filter=inv.payload.get("metadata_filter") or None,
    )
    return EvidenceOutput(bundle=bundle)


def default_nl2sql_runner(inv: BranchInvocation) -> SQLOutput:
    if inv.payload.get("intent") == "topic_claim_audit":
        direct = _topic_claim_audit_sql(inv)
        if direct is not None:
            return direct
    from tools.nl2sql_tools import NL2SQLTool
    return NL2SQLTool().answer(
        inv.payload.get("nl_query", ""),
        topic_id_hints=inv.payload.get("topic_id_hints"),
    )


def _topic_claim_audit_sql(inv: BranchInvocation) -> Optional[SQLOutput]:
    """Deterministic SQL for topic-scoped claim audits.

    The task needs source rows for claim extraction, not an aggregate. Keeping
    this path deterministic avoids NL2SQL drifting into counts or topic lists.
    """
    hints = inv.payload.get("topic_id_hints") or []
    topic_ids = [
        str(h.get("topic_id") or "")
        for h in hints
        if isinstance(h, dict) and h.get("topic_id")
    ]
    if not topic_ids:
        return None

    final_sql = (
        "SELECT p.post_id, p.author, p.subreddit, p.posted_at, "
        "p.like_count, p.reply_count, p.dominant_emotion, "
        "t.topic_id, t.label AS topic_label, p.text "
        "FROM posts_v2 p "
        "LEFT JOIN topics_v2 t ON p.topic_id = t.topic_id "
        "WHERE p.source = 'reddit' "
        f"AND p.topic_id IN ({', '.join(repr(t) for t in topic_ids)}) "
        "ORDER BY (p.like_count + p.reply_count) DESC, "
        "p.posted_at DESC NULLS LAST "
        "LIMIT 20"
    )

    out = SQLOutput(
        nl_query=inv.payload.get("nl_query", ""),
        final_sql=final_sql,
        columns=[
            "post_id", "author", "subreddit", "posted_at",
            "like_count", "reply_count", "dominant_emotion",
            "topic_id", "topic_label", "text",
        ],
    )
    try:
        from services.postgres_service import PostgresService
        pg = PostgresService()
        pg.connect()
        with pg.cursor() as cur:
            cur.execute(
                """
                SELECT p.post_id, p.author, p.subreddit, p.posted_at,
                       p.like_count, p.reply_count, p.dominant_emotion,
                       t.topic_id, t.label AS topic_label, p.text
                FROM posts_v2 p
                LEFT JOIN topics_v2 t ON p.topic_id = t.topic_id
                WHERE p.source = 'reddit'
                  AND p.topic_id = ANY(%s)
                ORDER BY (p.like_count + p.reply_count) DESC,
                         p.posted_at DESC NULLS LAST
                LIMIT 20
                """,
                (topic_ids,),
            )
            out.rows = list(cur.fetchall())
        out.success = True
    except Exception as exc:
        out.attempts.append(SQLAttempt(
            sql=final_sql,
            error=str(exc)[:240],
            error_kind="sql_connection",
        ))
        out.success = False
    return out


def _enrich_kg_output_with_entities(out: KGOutput, query_text: str) -> None:
    """Append Entity nodes for entities mentioned in the query.

    KG analytics returns Account/Post topology (IDs, PageRank, Louvain).
    Named entities (Iran, Russia, Trump …) won't appear unless we fetch them
    explicitly. Without this, the report writer never mentions entity names.
    """
    if not query_text:
        return
    try:
        from tools.kg_query_tools import KGQueryTool
        kuzu = KGQueryTool().kuzu
        if kuzu is None:
            return
        rows = kuzu._safe_execute(
            "MATCH (e:Entity) WHERE e.name IS NOT NULL "
            "RETURN e.id AS entity_id, e.name AS name, "
            "e.entity_type AS entity_type, e.mention_count AS mention_count "
            "ORDER BY e.mention_count DESC LIMIT 200",
            {},
        ) or []
    except Exception:
        return

    import re as _re
    query_lower = query_text.lower()
    existing_ids = {n.id for n in out.nodes}
    matched: list[str] = []
    for r in rows:
        name = (r.get("name") or "").strip()
        if not name:
            continue
        if _re.search(r"\b" + _re.escape(name.lower()) + r"\b", query_lower):
            eid = str(r.get("entity_id") or name)
            if eid not in existing_ids:
                out.nodes.append(KGNode(
                    id=eid,
                    label="Entity",
                    properties={
                        "name": name,
                        "entity_type": r.get("entity_type") or "",
                        "mention_count": int(r.get("mention_count") or 0),
                    },
                ))
                existing_ids.add(eid)
            matched.append(name)
    if matched:
        out.metrics["query_entities"] = matched


def default_kg_runner(inv: BranchInvocation) -> KGOutput:
    """Dispatch a KG branch invocation to the right tool.

    Routing (Phase C):
      propagation_trace   -> tools.kg_query_tools.propagation_path
      cascade_query       -> tools.kg_query_tools.viral_cascade
      influencer_query    -> agents.kg_analytics.influencer_rank (PageRank)
      coordination_check  -> agents.kg_analytics.coordinated_groups (Louvain)
      community_structure -> agents.kg_analytics.echo_chamber (modularity)
      propagation         -> coordinated_groups (legacy alias)
      anything else       -> influencer_rank as a sensible default
    """
    from tools.kg_query_tools import KGQueryTool

    intent = inv.payload.get("intent", "freeform")
    topic_id = inv.payload.get("topic_id") or ""
    topic_id = _coerce_topic_id(topic_id)
    topic_candidates = _candidate_topic_ids(inv, topic_id)

    # Anchor-less queries: try to back-fill from the most-discussed topic so
    # KG actually produces signal.
    if not topic_id and intent not in {
        "propagation_trace", "coordination_check",
    }:
        topic_id = _resolve_default_topic_id() or ""

    _query = inv.payload.get("query", "")

    # ── propagation_trace (KGQueryTool) ──────────────────────────────────────
    if intent == "propagation_trace":
        src = inv.payload.get("source_account") or ""
        dst = inv.payload.get("target_account") or ""
        if not src or not dst:
            if topic_candidates:
                out = _first_kg_output_with_signal(
                    topic_candidates,
                    lambda tid: KGQueryTool().topic_reply_chains(
                        topic_id=tid,
                        top_k=int(inv.payload.get("top_k", 5)),
                        max_depth=int(inv.payload.get("max_depth", 5)),
                    ),
                )
            else:
                from models.branch_output import KGOutput as _KGOut
                out = _KGOut(
                    query_kind="propagation_path",
                    target={"reason": "missing source/target account"},
                )
        else:
            out = KGQueryTool().propagation_path(
                source_account=src, target_account=dst,
                max_hops=int(inv.payload.get("max_hops", 6)),
            )
        _enrich_kg_output_with_entities(out, _query)
        return out

    # ── cascade_query (KGQueryTool) ──────────────────────────────────────────
    if intent == "cascade_query":
        query_text_lc = (inv.payload.get("query") or "").lower()
        if (
            "reply chain" in query_text_lc
            or "reply chains" in query_text_lc
            or "thread" in query_text_lc
            or "propagation path" in query_text_lc
            or "propagation paths" in query_text_lc
            or "spread" in query_text_lc
            or "dissemination" in query_text_lc
        ):
            out = _first_kg_output_with_signal(
                topic_candidates,
                lambda tid: KGQueryTool().topic_reply_chains(
                    topic_id=tid,
                    top_k=int(inv.payload.get("top_k", 5)),
                    max_depth=int(inv.payload.get("max_depth", 5)),
                ),
            )
        else:
            out = _first_kg_output_with_signal(
                topic_candidates,
                lambda tid: KGQueryTool().viral_cascade(
                    topic_id=tid,
                    top_k=int(inv.payload.get("top_k", 5)),
                ),
            )
        _enrich_kg_output_with_entities(out, _query)
        return out

    # ── KGAnalytics methods (Phase B.3) ──────────────────────────────────────
    from agents.kg_analytics import KGAnalytics
    analytics = KGAnalytics()

    # Day 7 freshness: trending / propagation / influencer queries
    # default to a 30-day window. Fact-check / official-recap / general
    # KG queries don't pre-filter by time.
    since_days = int(inv.payload.get("since_days", 30))

    if intent == "influencer_query":
        out = _first_kg_output_with_signal(
            topic_candidates or [""],
            lambda tid: analytics.influencer_rank(
                topic_id=tid or None,
                top_k=int(inv.payload.get("top_k", 10)),
                since_days=since_days,
            ),
        )
        _enrich_kg_output_with_entities(out, _query)
        return out
    if intent in ("coordination_check", "propagation"):
        out = _first_kg_output_with_signal(
            topic_candidates or [""],
            lambda tid: analytics.coordinated_groups(
                topic_id=tid or None,
                min_size=int(inv.payload.get("min_size", 3)),
                since_days=since_days,
            ),
        )
        _enrich_kg_output_with_entities(out, _query)
        return out
    if intent == "community_structure":
        out = _first_kg_output_with_signal(
            topic_candidates,
            lambda tid: analytics.echo_chamber(
                topic_id=tid,
                modularity_threshold=float(
                    inv.payload.get("modularity_threshold", 0.3),
                ),
            ),
        )
        _enrich_kg_output_with_entities(out, _query)
        return out

    # Fallback: PageRank gives a more useful default than raw post counts.
    # When no topic anchor is available, run global PageRank rather than
    # returning an empty stub - the previous behaviour produced 0/0 KG cards
    # in the UI even though the analytics layer supports topic_id=None.
    out = _first_kg_output_with_signal(
        topic_candidates or [""],
        lambda tid: analytics.influencer_rank(
            topic_id=tid or None, top_k=10, since_days=since_days,
        ),
    )
    _enrich_kg_output_with_entities(out, _query)
    return out


_ACCOUNT_ID_RE = re.compile(
    r"\b(?:u_|user_|@)?([a-zA-Z][a-zA-Z0-9_]{1,30})\b"
)


def _extract_account_ids(text: str) -> list[str]:
    """Best-effort scan of natural-language text for account ids.

    Looks for tokens shaped like `u_alice`, `@bob`, `user_carol`. Falls
    back to confirming each candidate against the live PG `posts_v2`
    table so we don't pick up generic English nouns. The check is cached
    cheap (single SELECT WHERE author IN (...)).
    """
    if not text:
        return []
    # Stage 1: token shapes that strongly imply an account id
    direct: list[str] = []
    for m in re.finditer(r"\b(u_[a-zA-Z][a-zA-Z0-9_]*|user_[a-zA-Z][a-zA-Z0-9_]*|@[a-zA-Z][a-zA-Z0-9_]*)\b", text):
        tok = m.group(1)
        if tok.startswith("@"):
            tok = tok[1:]
        direct.append(tok)
    if len(direct) >= 2:
        return direct[:5]

    # Stage 2: ask Postgres which words appear as authors. This catches
    # plain handles like "alice" without prefix.
    try:
        from services.postgres_service import PostgresService
        pg = PostgresService()
        pg.connect()
    except Exception:
        return direct
    candidates = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{1,30}\b", text)
    candidates = [c for c in candidates if len(c) >= 2]
    if not candidates:
        return direct
    try:
        with pg.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT account_id, author "
                "FROM posts_v2 "
                "WHERE LOWER(account_id) = ANY(%s) "
                "   OR LOWER(author) = ANY(%s)",
                ([c.lower() for c in candidates],
                 [c.lower() for c in candidates]),
            )
            rows = cur.fetchall() or []
    except Exception:
        return direct
    seen: set[str] = set(direct)
    out: list[str] = list(direct)
    # Preserve mention order from the text
    lower_to_account = {}
    for r in rows:
        lower_to_account[(r["author"] or "").lower()] = r["account_id"]
        lower_to_account[(r["account_id"] or "").lower()] = r["account_id"]
    for c in candidates:
        match = lower_to_account.get(c.lower())
        if match and match not in seen:
            seen.add(match)
            out.append(match)
    return out[:5]


def _looks_like_topic_id(value: str) -> bool:
    return bool(re.match(r"^topic_[A-Za-z0-9]+$", value or ""))


def _is_broad_topic_listing(sub: Subtask) -> bool:
    """True when the user wants the topic catalogue, not one topic."""
    text = (sub.text or "").lower()
    if sub.intent not in {"community_listing", "community_count", "trend"}:
        return False
    if not re.search(
        r"\b(list|show|what|which|all|today'?s|todays)\b.*\btopics?\b"
        r"|\btopics?\b.*\b(topic_id|post count|dominant emotion|label)\b"
        r"|有哪些\s*topics?",
        text,
    ):
        return False
    return not re.search(
        r"\b(amplif|influenc|propagat|spread|path|trace|cascade|central)\w*\b",
        text,
    )


def _subtask_needs_kg_signal(sub: Subtask) -> bool:
    if "kg" in (sub.suggested_branches or []):
        return True
    if sub.intent in {
        "propagation", "propagation_trace", "influencer_query",
        "coordination_check", "community_structure", "cascade_query",
        "explain_decision",
    }:
        return True
    text = (sub.text or "").lower()
    return bool(re.search(
        r"\b(knowledge graph|propagat|reply chains?|spread|path|trace|"
        r"cascade|amplif|influenc|centrality|network|echo chamber)\w*\b",
        text,
    ))


def _coerce_topic_id(value: str) -> str:
    """Return a real topic_id for `value`, resolving labels when needed."""
    value = (value or "").strip()
    if not value:
        return ""
    if _looks_like_topic_id(value):
        return value
    # Preserve compact synthetic ids used by tests and legacy fixtures. Natural
    # language labels from the LLM usually contain spaces or punctuation.
    if re.match(r"^[A-Za-z0-9_-]+$", value):
        return value
    try:
        from tools.topic_resolver import TopicResolver
        matches = TopicResolver().resolve(value, top_k=1)
        if matches:
            return matches[0].topic_id
    except Exception as exc:
        log.warning("planner_v2.topic_coerce_failed",
                    topic=value[:80], error=str(exc)[:120])
    return value


def _candidate_topic_ids(inv: BranchInvocation, primary_topic_id: str) -> list[str]:
    """Return ordered topic candidates for KG queries.

    TopicResolver may return several plausible topics. KG should not give up
    after the first candidate if that subgraph has no nodes/edges.
    """
    out: list[str] = []
    if primary_topic_id:
        out.append(primary_topic_id)
    for hint in inv.payload.get("topic_id_hints") or []:
        if not isinstance(hint, dict):
            continue
        tid = _coerce_topic_id(str(hint.get("topic_id") or ""))
        if tid and tid not in out:
            out.append(tid)
    return out


def _first_kg_output_with_signal(
    topic_ids: list[str],
    factory: Callable[[str], KGOutput],
) -> KGOutput:
    """Try KG topic candidates until one returns graph signal.

    This is a workflow-level guard against near-miss topic resolution:
    candidate #1 can be semantically close but have no Kuzu subgraph, while
    candidate #2/#3 has the actual reply/cascade evidence.
    """
    if not topic_ids:
        return KGOutput(
            query_kind="influencer_rank",
            target={"reason": "no topic anchor available"},
        )
    first: Optional[KGOutput] = None
    attempted: list[str] = []
    for tid in topic_ids:
        attempted.append(tid)
        out = factory(tid)
        out.target.setdefault("attempted_topic_ids", list(attempted))
        out.target.setdefault("selected_topic_id", tid)
        if first is None:
            first = out
        if _kg_output_has_signal(out):
            if attempted and tid != attempted[0]:
                out.target["fallback_from_topic_id"] = attempted[0]
                out.target["fallback_reason"] = (
                    "primary topic had no KG graph signal; selected a "
                    "semantically similar topic with graph evidence"
                )
                out.target["attempted_topic_ids"] = list(attempted)
            return out
    assert first is not None
    first.target.setdefault("attempted_topic_ids", attempted)
    return first


def _kg_output_has_signal(out: KGOutput) -> bool:
    if out.nodes or out.edges:
        return True
    for key, value in out.metrics.items():
        if key in {"cache_hit", "since_days"}:
            continue
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def _resolve_default_topic_id() -> str | None:
    """Best-effort: return the most-discussed topic_id from PG."""
    try:
        from services.postgres_service import PostgresService
        pg = PostgresService()
        pg.connect()
        with pg.cursor() as cur:
            cur.execute(
                "SELECT topic_id FROM topics_v2 "
                "ORDER BY post_count DESC LIMIT 1"
            )
            row = cur.fetchone()
            return row.get("topic_id") if row else None
    except Exception:
        return None
