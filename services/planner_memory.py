"""
Planner memory service - redesign-2026-05 Phase 3.5.

Wraps Chroma 3 (`chroma_planner`). Stores three flavours of documents:

    kind=module_card         - one per branch (evidence / nl2sql / kg)
    kind=workflow_success    - successful (question, branches_used) exemplars
    kind=workflow_error      - planner-attribution failures (Critic-driven)
    kind=composition_error   - report-writer failures (citation/numeric)

Conflict policy mirrors NL2SQL memory (PROJECT_REDESIGN_V2.md 7c-H, Q11=B):
    sim < 0.92                -> append
    0.92 <= sim < 0.95        -> direct overwrite
    sim >= 0.95               -> LLM-arbitrated comparison
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import structlog

from config import (
    NL2SQL_CONFLICT_SIM_HIGH,
    NL2SQL_CONFLICT_SIM_LOW,
)
from models.module_card import ModuleCard, WorkflowExemplar
from services.chroma_collections import ChromaCollections

log = structlog.get_logger(__name__)


PlannerKind = Literal[
    "module_card",
    "workflow_success",
    "workflow_error",
    "composition_error",
]


@dataclass
class PlannerMemory:
    collections: Optional[ChromaCollections] = None
    sim_low: float = NL2SQL_CONFLICT_SIM_LOW
    sim_high: float = NL2SQL_CONFLICT_SIM_HIGH
    llm_judge: Optional[Callable[[str, str], bool]] = None

    def __post_init__(self) -> None:
        self.collections = self.collections or ChromaCollections()
        self.llm_judge = self.llm_judge or (lambda new, old: True)

    # ── Module cards ──────────────────────────────────────────────────────────

    def upsert_module_card(self, card: ModuleCard, embedding: list[float]) -> str:
        record_id = f"module_card::{card.name}"
        meta = {
            "kind": "module_card",
            "branch": card.name,
            "updated_at": time.time(),
        }
        self.collections.planner.upsert(
            ids=[record_id],
            embeddings=[embedding],
            documents=[card.doc_text()],
            metadatas=[meta],
        )
        return record_id

    def upsert_workflow_success(
        self, exemplar: WorkflowExemplar, embedding: list[float],
    ) -> str:
        meta = {
            "kind": "workflow_success",
            "branches": ",".join(exemplar.branches_used),
            "confidence": 0.5,
            "hit_count": 0,
            "last_used_at": time.time(),
        }
        return self._upsert_with_conflict(exemplar.doc_text(), embedding, meta)

    def upsert_workflow_error(
        self,
        question: str,
        branches_used: list[str],
        error_kind: str,
        embedding: list[float],
    ) -> str:
        text = (f"Question: {question}\n"
                f"Branches: {', '.join(branches_used)}\n"
                f"Error: {error_kind}")
        meta = {
            "kind": "workflow_error",
            "error_kind": error_kind,
            "branches": ",".join(branches_used),
            "confidence": 0.5,
            "hit_count": 0,
            "last_used_at": time.time(),
        }
        return self._upsert_with_conflict(text, embedding, meta)

    def upsert_composition_error(
        self,
        question: str,
        error_kind: str,
        excerpt: str,
        embedding: list[float],
    ) -> str:
        text = (f"Question: {question}\nError: {error_kind}\n"
                f"Excerpt: {excerpt[:300]}")
        meta = {
            "kind": "composition_error",
            "error_kind": error_kind,
            "confidence": 0.5,
            "hit_count": 0,
            "last_used_at": time.time(),
        }
        return self._upsert_with_conflict(text, embedding, meta)

    # ── Recall ───────────────────────────────────────────────────────────────

    def recall_module_cards(
        self, embedding: list[float], n_results: int = 3,
    ) -> list[dict]:
        return self.collections.planner.query(
            embedding=embedding, n_results=n_results,
            where={"kind": "module_card"},
        )

    def recall_workflow_exemplars(
        self, embedding: list[float], n_results: int = 5,
    ) -> list[dict]:
        return self.collections.planner.query(
            embedding=embedding, n_results=n_results,
            where={"kind": "workflow_success"},
        )

    def recall_workflow_errors(
        self, embedding: list[float], n_results: int = 3,
    ) -> list[dict]:
        return self.collections.planner.query(
            embedding=embedding, n_results=n_results,
            where={"kind": "workflow_error"},
        )

    def recall_recent_route_violations(
        self, embedding: list[float], n_results: int = 5,
    ) -> list[dict]:
        """Recall verifier corrections (negative few-shot for the Rewriter).

        These are `workflow_error` records whose `error_kind` is
        `route_violation:<rule_id>` - written by ChatOrchestrator whenever
        PlanVerifier had to fix a routing decision. They serve as an
        anti-pattern channel: the next Rewriter call sees "last time you
        chose X for this kind of question, the verifier had to correct it".

        Deduplication: at most one record per rule_id is kept in the first
        pass, so the negative few-shot covers different failure modes
        instead of repeating the same one. Remaining slots are filled by
        next-best similarity regardless of rule_id.
        """
        # Chroma `$contains` only works on documents, so over-fetch and
        # post-filter in Python.
        raw = self.collections.planner.query(
            embedding=embedding,
            n_results=max(n_results * 4, n_results),
            where={"kind": "workflow_error"},
        )
        violations: list[tuple[str, dict]] = []
        for r in raw:
            meta = r.get("metadata") or {}
            kind = str(meta.get("error_kind") or "")
            if kind.startswith("route_violation:"):
                rule_id = kind.split(":", 1)[1] or "unknown"
                violations.append((rule_id, r))

        # Pass 1: one record per rule_id, in similarity order.
        seen_rules: set[str] = set()
        out: list[dict] = []
        for rule_id, rec in violations:
            if rule_id in seen_rules:
                continue
            seen_rules.add(rule_id)
            out.append(rec)
            if len(out) >= n_results:
                return out

        # Pass 2: fill any remaining slots with next-best regardless of rule.
        seen_ids = {r.get("id") for r in out}
        for _, rec in violations:
            if rec.get("id") in seen_ids:
                continue
            out.append(rec)
            if len(out) >= n_results:
                break
        return out

    def count_branch_combo_successes(self, branches_used: list[str]) -> int:
        """Used by Q9 confidence rule (PROJECT_REDESIGN_V2.md 7c-B)."""
        key = ",".join(branches_used)
        try:
            results = self.collections.planner.handle.get(
                where={"kind": "workflow_success"},
                include=["metadatas"],
            )
        except Exception as exc:
            log.warning("planner_memory.count_branch_combo_failed",
                        error=str(exc)[:160])
            return 0
        count = 0
        for meta in results.get("metadatas") or []:
            if (meta or {}).get("branches") == key:
                count += 1
        return count

    def prune_stale_topic_references(
        self,
        live_topic_ids: set[str],
    ) -> list[str]:
        """Delete Chroma 3 records that reference removed topic ids."""
        if not live_topic_ids:
            return []
        try:
            results = self.collections.planner.handle.get(
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            log.warning("planner_memory.prune_topics_get_failed",
                        error=str(exc)[:160])
            return []

        stale_ids: list[str] = []
        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        for idx, record_id in enumerate(ids):
            doc = docs[idx] if idx < len(docs) else ""
            meta = metas[idx] if idx < len(metas) else {}
            haystack = f"{doc}\n{meta}"
            refs = set(re.findall(r"\btopic_[0-9a-fA-F]{6,}\b", haystack))
            if refs and not refs.issubset(live_topic_ids):
                stale_ids.append(str(record_id))

        if stale_ids:
            self.collections.planner.delete(ids=stale_ids)
            log.info("planner_memory.pruned_stale_topics",
                     count=len(stale_ids),
                     ids=stale_ids[:20])
        return stale_ids

    def delete_records(self, record_ids: list[str]) -> None:
        if record_ids:
            self.collections.planner.delete(ids=record_ids)

    # ── Internals ────────────────────────────────────────────────────────────

    def _upsert_with_conflict(
        self, text: str, embedding: list[float], metadata: dict,
    ) -> str:
        kind = metadata.get("kind", "")
        existing = self.collections.planner.query(
            embedding=embedding, n_results=5, where={"kind": kind},
        )
        if existing:
            top = existing[0]
            sim = float(top.get("similarity", 0.0))
            old_id = top["id"]
            old_text = top.get("document", "")
            if sim >= self.sim_high:
                if self.llm_judge(text, old_text):
                    self.collections.planner.delete(ids=[old_id])
            elif sim >= self.sim_low:
                self.collections.planner.delete(ids=[old_id])
        record_id = f"{kind}::{uuid.uuid4().hex}"
        self.collections.planner.upsert(
            ids=[record_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return record_id


# ── Seed cards (PROJECT_REDESIGN_V2.md Phase 2 5b cold-start) ────────────────

SEED_MODULE_CARDS: list[ModuleCard] = [
    ModuleCard(
        name="evidence",
        description=("Hybrid (dense + BM25 + RRF + rerank) retrieval over "
                      "Chroma 1, the official-sources collection. Use ONLY "
                      "when the user explicitly wants authoritative reporting "
                      "compared to or surfaced for the community side."),
        when_to_use=[
            "User asks for fact-check evidence from authoritative outlets.",
            "User wants to compare community claims against official reporting.",
            "Question explicitly mentions sources like BBC / NYT / Reuters / Wikipedia.",
        ],
        when_not_to_use=[
            "Pure structural questions about who posted / how often.",
            "Questions about graph relationships between accounts.",
            "Listing claims / topics / community emotion inside the Reddit "
            "corpus — those questions are answered by nl2sql alone, even "
            "when the user mentions claims. Adding evidence here just "
            "fans out a wasted official-source query.",
            "Anything that does not explicitly need authoritative comparison.",
        ],
        input_schema={
            "query": "string",
            "metadata_filter": "dict (optional, e.g. {'tier': 'reputable_media'})",
        },
        output_schema={"bundle": "EvidenceBundle (chunks + citations + ranks)"},
        examples=[
            {"question": "Did the BBC report on the vaccine recall last week?"},
            {"question": "What is the WHO's official position on long COVID?"},
        ],
    ),
    ModuleCard(
        name="nl2sql",
        description=("Generates and executes a safe read-only Postgres "
                      "SELECT against posts_v2 / topics_v2 / entities_v2 / "
                      "claims_v2 / post_claims_v2. Owns the COMMUNITY side: "
                      "what was said, who said it, how often, and which "
                      "atomic claims they discussed. Fact-check / claim-"
                      "audit queries on Reddit data should land here, not "
                      "on evidence — claims_v2 stores the canonical claim "
                      "text and a hybrid claim_search (cosine + simhash + "
                      "tsvector) finds matching claims even when the user's "
                      "phrasing diverges from the original."),
        when_to_use=[
            "Counting / filtering / grouping over community posts.",
            "Topic-scoped questions ('show me posts in topic T').",
            "Time-window queries on posted_at.",
            "Listing claims in a topic (read claims_v2; do not regenerate "
            "claim text from posts_v2).",
            "Fact-checking a Reddit claim against the community corpus "
            "(use claim_search via claim_id_hints).",
        ],
        when_not_to_use=[
            "Questions about graph paths or centrality - use the kg branch.",
            "Pulling official-news evidence (BBC / NYT / Reuters / AP / "
            "Xinhua) - that's the evidence branch over chroma_official.",
        ],
        input_schema={
            "nl_query": "string",
            "topic_id_hints": "list of {topic_id,label,similarity} (optional)",
            "claim_id_hints": "list of {claim_id,claim_text,score} from "
                               "hybrid claim_search (optional)",
        },
        output_schema={"sql_output": "SQLOutput (final_sql, rows, attempts)"},
        examples=[
            {"question": "How many posts in topic T have dominant_emotion='anger'?"},
            {"question": "List the top 10 authors in the last 7 days."},
            {"question": "List the claims discussed in topic T with author counts."},
            {"question": "Which Reddit claims about the Iran ceasefire match "
                          "the BBC story (semantic match via claim_search)?"},
        ],
    ),
    ModuleCard(
        name="kg",
        description=(
            "The ONLY branch that can do multi-hop reply traversal, "
            "centrality (PageRank / betweenness), and community detection "
            "(Louvain) over the Kuzu graph. SQL cannot express any of "
            "these - routing them to nl2sql produces shallow GROUP BY "
            "answers that miss the structure of the spread."
        ),
        when_to_use=[
            "Tracing a reply chain between two accounts (propagation_trace).",
            "Identifying influence by PageRank, not raw post counts "
            "(influencer_query).",
            "Detecting coordinated posting / bot networks "
            "(coordination_check, Louvain communities).",
            "Diagnosing echo chambers / polarised clusters "
            "(community_structure, modularity).",
            "Surfacing viral cascades / longest reply threads "
            "(cascade_query).",
            "Answering 'who is amplifying' / 'how did this spread' / "
            "'are they organised'.",
        ],
        when_not_to_use=[
            "Simple counts or filters that don't need graph structure - "
            "use nl2sql.",
            "Verifying facts against external authoritative reports - "
            "use evidence.",
            "Listing the text of posts in a topic - nl2sql is faster.",
        ],
        input_schema={
            "query_kind": (
                "propagation_path | key_nodes | topic_correlation | "
                "cascade_tree | viral_cascade | influencer_rank | "
                "bridge_accounts | coordinated_groups | echo_chamber"
            ),
            "target": (
                "dict (topic_id / source_account / target_account / "
                "root_post_id / top_k / min_size)"
            ),
        },
        output_schema={"kg_output": "KGOutput (nodes, edges, metrics)"},
        examples=[
            {"question": "Show me how the vaccine rumour spread from "
                          "alice to dave."},
            {"question": "Who's the most influential account in the "
                          "climate topic? (rank by PageRank, not post count)"},
            {"question": "Is the surge of anti-vaccine posts coming from "
                          "a coordinated group?"},
            {"question": "Is topic T an echo chamber?"},
            {"question": "What's the deepest reply thread under any "
                          "post in this topic?"},
            {"question": "Find the bridge accounts that connect the "
                          "vaccine cluster and the climate cluster."},
        ],
    ),
]


SEED_WORKFLOW_EXEMPLARS: list[WorkflowExemplar] = [
    WorkflowExemplar(
        question="Is the BBC story about the vaccine recall being amplified on Reddit?",
        branches_used=["evidence", "nl2sql", "kg"],
        rationale="Need official source (evidence), Reddit volume (nl2sql), "
                  "amplifier accounts (kg).",
    ),
    WorkflowExemplar(
        question="What topics are trending right now?",
        branches_used=["nl2sql", "kg"],
        rationale="Trending = volume by topic (nl2sql) + the amplifier "
                  "structure behind each topic (kg).",
    ),
    WorkflowExemplar(
        question="List the claims in the topic about the Iran ceasefire.",
        branches_used=["nl2sql"],
        rationale="Pure community-side enumeration. claims_v2 + "
                  "post_claims_v2 already stores the canonical claim text; "
                  "do NOT route to evidence — official sources are not the "
                  "answer to 'list claims', and adding the evidence branch "
                  "wastes a call.",
    ),
    WorkflowExemplar(
        question="What is the dominant emotion in the topic about climate?",
        branches_used=["nl2sql"],
        rationale="Single-aggregation question over posts_v2. Evidence and "
                  "kg add no signal; route to nl2sql only.",
    ),
    WorkflowExemplar(
        question="Fact-check this Reddit claim about the Victory Day parade "
                  "against official sources.",
        branches_used=["evidence", "nl2sql"],
        rationale="Explicit verification: evidence pulls authoritative "
                  "reporting, nl2sql + claim_search finds the matching "
                  "claims_v2 row(s) for the community side.",
    ),
    WorkflowExemplar(
        question="Who is the most influential account in topic T?",
        branches_used=["kg", "nl2sql"],
        rationale="Centrality from graph + post counts as a tiebreaker.",
    ),
    WorkflowExemplar(
        question="What did Reuters say about the WHO statement, and how is the community reacting?",
        branches_used=["evidence", "nl2sql"],
        rationale="Authoritative recap plus community-side commentary.",
    ),
    WorkflowExemplar(
        question="How many angry posts about climate in the past week?",
        branches_used=["nl2sql"],
        rationale="Pure single-aggregation SQL job; no other branch helps.",
    ),
    WorkflowExemplar(
        question="Compare the official line and community sentiment on the vaccine recall.",
        branches_used=["evidence", "nl2sql", "kg"],
        rationale="Canonical 3-source comparison: official + volume + propagation.",
    ),
    WorkflowExemplar(
        question="Are accounts in topic T connected to topic U through shared entities?",
        branches_used=["kg", "nl2sql"],
        rationale="Cross-topic graph correlation, with NL2SQL providing "
                  "topic-level context (post counts, dominant emotion).",
    ),
    WorkflowExemplar(
        question="What's the dominant narrative in the current trending topic?",
        branches_used=["nl2sql", "kg", "evidence"],
        rationale="Topic content (nl2sql) + amplifiers (kg) + whether "
                  "official sources back it up (evidence).",
    ),
    # ── Phase C: KG-specialised exemplars ────────────────────────────────────
    WorkflowExemplar(
        question="Trace how the rumour spread from alice to dave through replies.",
        branches_used=["kg"],
        rationale="propagation_trace: multi-hop reply chain - "
                  "SQL cannot express it.",
    ),
    WorkflowExemplar(
        question="Who is most influential in the vaccine topic?",
        branches_used=["kg", "nl2sql"],
        rationale="influencer_query: PageRank from KG (real influence), "
                  "post counts from nl2sql for context.",
    ),
    WorkflowExemplar(
        question="Is this surge of posts coming from a coordinated group?",
        branches_used=["kg"],
        rationale="coordination_check: Louvain communities; SQL self-joins "
                  "cannot do modularity optimisation.",
    ),
    WorkflowExemplar(
        question="Is topic T an echo chamber?",
        branches_used=["kg", "nl2sql"],
        rationale="community_structure: KG modularity score + nl2sql for "
                  "the within-cluster post sample.",
    ),
    WorkflowExemplar(
        question="Show me the longest viral cascade in the climate topic.",
        branches_used=["kg"],
        rationale="cascade_query: viral_cascade ranks reply-tree depth; "
                  "SQL cannot follow recursive replies.",
    ),
]
