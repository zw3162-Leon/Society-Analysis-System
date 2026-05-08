"""
Precompute Pipeline v2 - redesign-2026-05.

Phase 1 backbone (5 stages):
    1. fetch_posts        - Fetch posts (Reddit / fixture)
    2. ingest             - Multimodal + entity enrichment + dedup
    3. normalize          - Field normalisation
    4. emotion_baseline   - Per-post baseline emotion classification
    5. topic_cluster      - Post-level embedding clustering

Phase 2 additions (this file):
    - schema_propose      - Schema-aware Agent emits SchemaProposal,
                            double-writes to PG schema_meta + Chroma 2
    - persist_v2          - Write posts/topics/entities to PG v2 tables and
                            Kuzu v2 relations

Optional dependencies are injected; when storage layers are unavailable
(test environments, missing services), the pipeline degrades gracefully:
each stage records a "degraded" status in the manifest instead of raising.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from models.post import Post

log = structlog.get_logger(__name__)


# ── Run artifact contract ────────────────────────────────────────────────────

@dataclass
class StageRecord:
    name: str
    status: str  # "ok" | "skipped" | "error" | "degraded"
    detail: str = ""
    elapsed_ms: int = 0


@dataclass
class PipelineV2Result:
    run_id: str
    run_dir: Path
    posts: list[Post] = field(default_factory=list)
    topics: list[Any] = field(default_factory=list)  # list[TopicCluster]
    duplicate_post_ids: set[str] = field(default_factory=set)
    schema_fingerprint: Optional[str] = None
    stages: list[StageRecord] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "v2",
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "post_count": len(self.posts),
            "topic_count": len(self.topics),
            "duplicate_post_count": len(self.duplicate_post_ids),
            "schema_fingerprint": self.schema_fingerprint,
            "stages": [
                {"name": s.name, "status": s.status, "detail": s.detail,
                 "elapsed_ms": s.elapsed_ms}
                for s in self.stages
            ],
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PrecomputePipelineV2:
    STAGES = [
        "fetch_posts",
        "ingest",
        "normalize",
        "emotion_baseline",
        "topic_cluster",
        "extract_claims",
        "schema_propose",
        "persist_v2",
    ]

    def __init__(
        self,
        ingestion,
        knowledge,
        multimodal=None,         # agents.multimodal_agent.MultimodalAgent
        entity_extractor=None,   # agents.entity_extractor.EntityExtractor
        topic_clusterer=None,    # agents.topic_clusterer.TopicClusterer
        post_deduper=None,       # agents.post_dedup.PostDeduper
        schema_agent=None,       # agents.schema_agent.SchemaAgent
        schema_sync=None,        # services.schema_sync.SchemaSync
        claim_extractor=None,    # agents.claim_extractor.ClaimExtractor
        pg=None,                 # services.postgres_service.PostgresService
        kuzu=None,               # services.kuzu_service.KuzuService
    ) -> None:
        self._ingestion = ingestion
        self._knowledge = knowledge
        self._multimodal = multimodal
        self._entity_extractor = entity_extractor
        self._topic_clusterer = topic_clusterer
        self._post_deduper = post_deduper
        self._schema_agent = schema_agent
        self._schema_sync = schema_sync
        self._claim_extractor = claim_extractor
        self._pg = pg
        self._kuzu = kuzu
        self._claim_result = None  # populated in stage 6

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(
        self,
        *,
        run_dir: Path,
        subreddits: Optional[list[str]] = None,
        reddit_query: Optional[str] = None,
        reddit_days_back: int = 3,
        reddit_limit_per_sub: int = 50,
        reddit_include_comments: bool = True,
        reddit_comment_limit: int = 100,
        jsonl_path: Optional[str] = None,
        claims_from: Optional[str] = None,
    ) -> PipelineV2Result:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_dir.name
        result = PipelineV2Result(run_id=run_id, run_dir=run_dir)

        # 1 - fetch_posts
        posts = self._stage(
            result, "fetch_posts",
            lambda: self._fetch(
                subreddits=subreddits,
                reddit_query=reddit_query,
                reddit_days_back=reddit_days_back,
                reddit_limit_per_sub=reddit_limit_per_sub,
                reddit_include_comments=reddit_include_comments,
                reddit_comment_limit=reddit_comment_limit,
                jsonl_path=jsonl_path or claims_from,
            ),
        ) or []

        # 2 - ingest (multimodal + entity + simhash)
        self._stage(result, "ingest", lambda: self._ingest(posts, result))

        # 3 - normalize
        self._stage(result, "normalize", lambda: self._normalize(posts))
        result.posts = posts

        # 4 - emotion_baseline
        self._stage(result, "emotion_baseline",
                    lambda: self._knowledge.classify_post_emotions(posts))

        # 5 - topic_cluster
        topics = self._stage(
            result, "topic_cluster", lambda: self._cluster_posts(posts),
        ) or []
        result.topics = topics

        # 6 - extract_claims (atomic claims from submission titles + comment links)
        self._stage(result, "extract_claims",
                    lambda: self._extract_claims(posts))

        # 7 - schema_propose (Phase 2.2 + 2.4 double-write)
        self._stage(result, "schema_propose",
                    lambda: self._propose_schema(posts, run_id, result))

        # 8 - persist_v2 (Phase 2.5 + 2.6 PG + Kuzu writes)
        self._stage(result, "persist_v2",
                    lambda: self._persist_v2(posts, topics, result))

        # Manifest
        manifest = result.to_manifest()
        (run_dir / "run_manifest_v2.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("pipeline_v2.done", run_id=run_id,
                 posts=len(posts), topics=len(topics),
                 duplicates=len(result.duplicate_post_ids))
        return result

    # ── Stage implementations ────────────────────────────────────────────────

    def _fetch(
        self, *, subreddits, reddit_query, reddit_days_back,
        reddit_limit_per_sub, reddit_include_comments,
        reddit_comment_limit,
        jsonl_path,
    ) -> list[Post]:
        if jsonl_path:
            return self._ingestion.ingest_posts_from_jsonl(jsonl_path)
        if reddit_query and subreddits:
            return self._ingestion.ingest_reddit_search(
                reddit_query, subreddits=subreddits, days_back=reddit_days_back,
            )
        if reddit_query:
            return self._ingestion.ingest_reddit_search(
                reddit_query, days_back=reddit_days_back,
            )
        if subreddits and len(subreddits) == 1:
            return self._ingestion.ingest_subreddit(
                subreddits[0],
                days_back=reddit_days_back,
                limit=reddit_limit_per_sub,
                include_comments=reddit_include_comments,
                comment_limit=reddit_comment_limit,
            )
        if subreddits:
            return self._ingestion.ingest_multi_subreddit(
                subreddits=subreddits,
                days_back=reddit_days_back,
                limit_per_sub=reddit_limit_per_sub,
                include_comments=reddit_include_comments,
                comment_limit=reddit_comment_limit,
            )
        log.warning("pipeline_v2.fetch.no_source")
        return []

    def _ingest(self, posts: list[Post], result: PipelineV2Result) -> None:
        if self._multimodal is not None:
            self._multimodal.enrich_posts(posts)
        if self._entity_extractor is not None:
            self._entity_extractor.extract_for_posts(posts)
        if self._post_deduper is not None:
            self._post_deduper.annotate(posts)
            report = self._post_deduper.find_duplicates(posts)
            result.duplicate_post_ids = report.duplicate_post_ids

    @staticmethod
    def _normalize(posts: list[Post]) -> None:
        for p in posts:
            if not p.id:
                p.id = f"post_{uuid.uuid4().hex[:10]}"
            if not p.account_id:
                p.account_id = "unknown"
            if p.text is None:
                p.text = ""

    def _cluster_posts(self, posts: list[Post]) -> list[Any]:
        if self._topic_clusterer is None or not posts:
            return []
        return self._topic_clusterer.cluster(posts)

    def _extract_claims(self, posts: list[Post]) -> Any:
        if self._claim_extractor is None or not posts:
            self._claim_result = None
            return None
        self._claim_result = self._claim_extractor.extract(posts)
        return self._claim_result

    def _propose_schema(
        self, posts: list[Post], run_id: str, result: PipelineV2Result,
    ) -> Optional[str]:
        if self._schema_agent is None or self._schema_sync is None:
            return None
        proposal = self._schema_agent.propose(run_id=run_id, posts=posts)
        try:
            self._schema_sync.apply_proposal(proposal)
            result.schema_fingerprint = proposal.schema_fingerprint()
            return result.schema_fingerprint
        except Exception as exc:
            log.error("pipeline_v2.schema_sync_error", error=str(exc)[:160])
            raise

    def _persist_v2(
        self, posts: list[Post], topics: list[Any], result: PipelineV2Result,
    ) -> None:
        if self._pg is None:
            log.info("pipeline_v2.persist_v2.no_pg")
            return

        skip_ids = result.duplicate_post_ids
        run_id = result.run_id
        for p in posts:
            if p.id in skip_ids:
                continue
            try:
                source, subreddit, author = _post_source_fields(p)
                self._pg.upsert_post_v2(
                    post_id=p.id,
                    account_id=p.account_id,
                    author=author,
                    text=p.text or "",
                    posted_at=p.posted_at,
                    subreddit=subreddit,
                    source=source,
                    topic_id=p.topic_id,
                    dominant_emotion=p.emotion or None,
                    emotion_score=p.emotion_score or 0.0,
                    like_count=int(p.like_count or 0),
                    reply_count=int(p.reply_count or 0),
                    retweet_count=int(p.retweet_count or 0),
                    simhash=p.simhash,
                    extra={},
                    run_id=run_id,
                )
            except Exception as exc:
                log.error("pipeline_v2.upsert_post_v2_error",
                          post_id=p.id, error=str(exc)[:120])

        # Topics
        for cluster in topics:
            try:
                self._pg.upsert_topic_v2(
                    topic_id=cluster.topic_id,
                    label=cluster.label,
                    post_count=cluster.post_count(),
                    dominant_emotion=cluster.dominant_emotion,
                    centroid_text=cluster.centroid_text,
                    extra={},
                    run_id=run_id,
                )
            except Exception as exc:
                log.error("pipeline_v2.upsert_topic_v2_error",
                          topic_id=cluster.topic_id, error=str(exc)[:120])

        # Entities (per post)
        for p in posts:
            if p.id in skip_ids:
                continue
            for span in getattr(p, "entities", []) or []:
                ent_id = f"ent_{uuid.uuid5(uuid.NAMESPACE_DNS, span.name.lower() + span.entity_type).hex[:12]}"
                try:
                    self._pg.upsert_entity_v2(
                        entity_id=ent_id,
                        name=span.name,
                        entity_type=span.entity_type,
                        mention_count=1,
                        run_id=run_id,
                    )
                    self._pg.link_post_entity_v2(
                        post_id=p.id,
                        entity_id=ent_id,
                        char_start=span.char_start,
                        char_end=span.char_end,
                        confidence=span.confidence,
                    )
                except Exception as exc:
                    log.error("pipeline_v2.entity_upsert_error",
                              entity=span.name, error=str(exc)[:120])

        # Claims (atomic claim extraction; M:N to posts)
        if self._claim_result is not None:
            for claim in self._claim_result.claims:
                try:
                    self._pg.upsert_claim_v2(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        topic_id=claim.topic_id,
                        embedding=claim.embedding,
                        simhash=claim.simhash,
                        source_url=claim.source_url,
                        role=claim.role,
                        confidence=claim.confidence,
                        run_id=run_id,
                    )
                except Exception as exc:
                    log.error("pipeline_v2.upsert_claim_v2_error",
                              claim_id=claim.claim_id, error=str(exc)[:160])
            for link in self._claim_result.links:
                if link.post_id in skip_ids:
                    continue
                try:
                    self._pg.link_post_claim_v2(
                        post_id=link.post_id,
                        claim_id=link.claim_id,
                        role=link.role,
                    )
                except Exception as exc:
                    log.error("pipeline_v2.link_post_claim_v2_error",
                              post_id=link.post_id,
                              claim_id=link.claim_id,
                              error=str(exc)[:160])

        # Kuzu writes - production hardening Day 6: collect everything
        # first, then issue per-table batches via the bulk_* APIs. This
        # halves wall-clock time vs the previous "interleave per-post"
        # pattern at typical (50-1000 post) batch sizes.
        if self._kuzu is not None:
            try:
                # Topics first so post -> topic edges find a target.
                for cluster in topics:
                    try:
                        self._kuzu.upsert_topic(cluster.topic_id, cluster.label)
                    except Exception:
                        pass

                accounts: list[tuple[str, str]] = []
                post_rows: list[tuple[str, str]] = []
                posted_rows: list[tuple[str, str]] = []
                replied_rows: list[tuple[str, str]] = []
                topic_rows: list[tuple[str, str]] = []
                placeholder_post_ids: set[str] = set()
                entity_pairs: list[tuple[str, str, str]] = []   # (eid, name, type)
                post_entity_rows: list[tuple[str, str]] = []

                for p in posts:
                    if p.id in skip_ids:
                        continue
                    accounts.append(
                        (p.account_id, p.channel_name or p.account_id)
                    )
                    post_rows.append((p.id, p.text or ""))
                    posted_rows.append((p.account_id, p.id))
                    if p.topic_id:
                        topic_rows.append((p.id, p.topic_id))
                    parent_id = getattr(p, "parent_post_id", None)
                    if parent_id:
                        if parent_id not in placeholder_post_ids:
                            placeholder_post_ids.add(parent_id)
                            post_rows.append((parent_id, ""))
                        replied_rows.append((p.id, parent_id))
                    for span in getattr(p, "entities", []) or []:
                        ent_id = f"ent_{uuid.uuid5(uuid.NAMESPACE_DNS, span.name.lower() + span.entity_type).hex[:12]}"
                        entity_pairs.append((ent_id, span.name, span.entity_type))
                        post_entity_rows.append((p.id, ent_id))

                # Order matters: nodes -> edges (FK-safe).
                self._kuzu.bulk_upsert_accounts(accounts)
                self._kuzu.bulk_upsert_posts(post_rows)
                for ent_id, name, etype in entity_pairs:
                    try:
                        self._kuzu.upsert_entity(ent_id, name, etype)
                    except Exception:
                        pass
                self._kuzu.bulk_add_posted(posted_rows)
                self._kuzu.bulk_add_belongs_to_topic(topic_rows)
                self._kuzu.bulk_add_replied(replied_rows)
                for pid, ent_id in post_entity_rows:
                    try:
                        self._kuzu.add_post_has_entity(pid, ent_id)
                    except Exception:
                        pass
                log.info("pipeline_v2.kuzu_bulk_done",
                         posts=len(post_rows),
                         accounts=len(accounts),
                         posted=len(posted_rows),
                         topic_edges=len(topic_rows),
                         reply_edges=len(replied_rows),
                         entities=len(entity_pairs),
                         entity_edges=len(post_entity_rows))
            except Exception as exc:
                log.error("pipeline_v2.kuzu_bulk_error",
                          error=str(exc)[:160])

        # redesign-2026-05-kg Phase C.4: invalidate the in-memory KG
        # subgraph cache so subsequent analytics queries see fresh edges.
        try:
            from services.kg_cache import bump_write_seq
            bump_write_seq()
        except Exception:
            pass

    # ── Stage helper ──────────────────────────────────────────────────────────

    def _stage(self, result: PipelineV2Result, name: str, fn) -> Any:
        t0 = datetime.now()
        try:
            out = fn()
            elapsed = int((datetime.now() - t0).total_seconds() * 1000)
            result.stages.append(StageRecord(
                name=name, status="ok", elapsed_ms=elapsed,
                detail=_summarize(out),
            ))
            return out
        except Exception as exc:
            elapsed = int((datetime.now() - t0).total_seconds() * 1000)
            log.error(f"pipeline_v2.{name}.error", error=str(exc))
            result.stages.append(StageRecord(
                name=name, status="error", elapsed_ms=elapsed,
                detail=f"{type(exc).__name__}: {exc}",
            ))
            return None


def _summarize(out: Any) -> str:
    if out is None:
        return ""
    if isinstance(out, list):
        return f"len={len(out)}"
    if isinstance(out, set):
        return f"set_len={len(out)}"
    if isinstance(out, str):
        return out[:60]
    return type(out).__name__


def _post_source_fields(post: Post) -> tuple[str, Optional[str], str]:
    """Normalize source metadata for the Reddit-only pipeline."""
    source = (post.source or "reddit").strip().lower()
    if source != "reddit":
        source = "reddit"

    subreddit = post.subreddit
    channel = (post.channel_name or "").strip()
    if not subreddit and channel.lower().startswith("r/"):
        subreddit = channel[2:]
    if subreddit:
        subreddit = subreddit.strip().lstrip("r/")

    return source, subreddit or None, post.account_id
