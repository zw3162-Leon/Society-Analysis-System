"""
Claim search — hybrid retrieval over claims_v2 (Reddit/community side).

Three independent ranked lists, fused with Reciprocal Rank Fusion (RRF, k=60):
  - Dense cosine via pgvector (`embedding <=> query`)
  - Simhash hamming distance (paraphrase / near-duplicate detection)
  - Postgres tsvector (`claim_text_tsv @@ plainto_tsquery`)

This is the community-side counterpart to `tools/hybrid_retrieval.py`, which
searches official-source chunks in Chroma 1. NL2SQL prompts should call this
tool for fact-check / topic-claim-audit queries instead of writing
`text_tsv @@ plainto_tsquery` against `posts_v2`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

from agents.post_dedup import compute_simhash
from services.embeddings_service import EmbeddingsService
from services.postgres_service import PostgresService

log = structlog.get_logger(__name__)


@dataclass
class ClaimHit:
    claim_id: str
    claim_text: str
    topic_id: Optional[str]
    source_url: Optional[str]
    role: Optional[str]
    confidence: float
    score: float = 0.0     # fused RRF score
    posts: list[dict] = field(default_factory=list)  # populated on demand


class ClaimSearchTool:
    def __init__(
        self,
        pg: Optional[PostgresService] = None,
        embeddings: Optional[EmbeddingsService] = None,
        *,
        rrf_k: int = 60,
        per_branch_limit: int = 30,
    ) -> None:
        self._pg = pg or PostgresService()
        self._embeddings = embeddings or EmbeddingsService()
        self._rrf_k = rrf_k
        self._per_branch_limit = per_branch_limit

    # ── Public API ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        topic_id: Optional[str] = None,
        top_k: int = 10,
        include_posts: bool = False,
    ) -> list[ClaimHit]:
        """Hybrid claim search. Returns top-k ClaimHits ranked by RRF."""
        query = (query or "").strip()
        if not query:
            return []

        try:
            embedding = self._embeddings.embed(query)
        except Exception as exc:
            log.warning("claim_search.embed_failed", error=str(exc)[:160])
            embedding = None
        simhash = compute_simhash(query)

        bundles = self._pg.search_claims_hybrid(
            query_text=query,
            query_embedding=embedding,
            query_simhash=simhash,
            topic_id=topic_id,
            limit=self._per_branch_limit,
        )

        ranks: dict[str, dict[str, int]] = {}
        rows_by_id: dict[str, dict] = {}
        for branch, rows in bundles.items():
            for rank, row in enumerate(rows):
                cid = row["claim_id"]
                rows_by_id.setdefault(cid, row)
                ranks.setdefault(cid, {})[branch] = rank

        fused: list[tuple[str, float]] = []
        for cid, branch_ranks in ranks.items():
            score = 0.0
            for r in branch_ranks.values():
                score += 1.0 / (self._rrf_k + r)
            fused.append((cid, score))
        fused.sort(key=lambda x: x[1], reverse=True)

        hits: list[ClaimHit] = []
        for cid, score in fused[:top_k]:
            row = rows_by_id[cid]
            hits.append(ClaimHit(
                claim_id=row["claim_id"],
                claim_text=row["claim_text"],
                topic_id=row.get("topic_id"),
                source_url=row.get("source_url"),
                role=row.get("role"),
                confidence=float(row.get("confidence") or 0.0),
                score=score,
            ))
        if include_posts:
            self._attach_posts(hits)
        log.info("claim_search.done",
                 query_len=len(query), topic_id=topic_id,
                 cosine=len(bundles.get("cosine", [])),
                 simhash=len(bundles.get("simhash", [])),
                 tsv=len(bundles.get("tsv", [])),
                 returned=len(hits))
        return hits

    def list_for_topic(
        self, topic_id: str, *, limit: int = 50,
    ) -> list[dict]:
        """Direct listing — bypass hybrid search, return all claims for a topic."""
        return self._pg.list_claims_for_topic(topic_id, limit=limit)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _attach_posts(self, hits: list[ClaimHit]) -> None:
        if not hits:
            return
        ids = [h.claim_id for h in hits]
        with self._pg.cursor() as cur:
            cur.execute(
                """
                SELECT pc.claim_id, pc.post_id, pc.role,
                       p.author, p.subreddit, p.posted_at,
                       LEFT(p.text, 280) AS text_excerpt
                FROM post_claims_v2 pc
                JOIN posts_v2 p USING (post_id)
                WHERE pc.claim_id = ANY(%s)
                ORDER BY p.posted_at DESC NULLS LAST
                """,
                (ids,),
            )
            rows = list(cur.fetchall())
        by_id: dict[str, list[dict]] = {}
        for r in rows:
            by_id.setdefault(r["claim_id"], []).append(dict(r))
        for h in hits:
            h.posts = by_id.get(h.claim_id, [])
