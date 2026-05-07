"""
Hybrid Retrieval - redesign-2026-05 Phase 3.2.

Branch A (Evidence Retrieval). Pipeline:

    Metadata pre-filter
        -> Dense recall (Chroma 1, cosine)
        -> BM25 recall (rank_bm25 in-memory over the same corpus subset)
        -> Reciprocal Rank Fusion (RRF, k=60)
        -> Optional rerank (bge-reranker-base, local; degrades to no-op
           when the package is missing)
        -> EvidenceBundle

The retrieval target is Chroma 1 (`chroma_official`). Posts are NOT
vectorised; community-side evidence flows through the NL2SQL branch's
tsvector + pg_trgm path instead.

Performance target (PROJECT_REDESIGN_V2.md Phase 3): top-50 recall + rerank
within ~2 s. The BM25 corpus is rebuilt on the metadata-filtered subset to
keep latency bounded. Production deployments may want to keep BM25 indexes
warm; that's a Phase 4 concern.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from models.evidence import Citation, EvidenceBundle, EvidenceChunk
from services.chroma_collections import ChromaCollections
from services.embeddings_service import EmbeddingsService

log = structlog.get_logger(__name__)

# Valid metadata keys for Chroma pre-filtering. publish_date is intentionally
# excluded: date-range strings from the UI cannot be expressed as a single
# Chroma operator, and hard date cuts silently drop semantically relevant
# articles. Date filtering belongs in the NL2SQL (SQL WHERE) branch.
_VALID_FILTER_KEYS = {"source", "domain", "tier", "title", "topic_hint"}
# LLM aliases → canonical key
_KEY_ALIASES = {
    "official_sources": "source",
    "sources":          "source",
    "news_source":      "source",
    "media_source":     "source",
}
# Keys that are never valid for the official-chunk store
_COMMUNITY_KEYS = {"subreddit", "author", "account_id", "post_id",
                   "publish_date", "date_range", "official_date_range"}


def _normalize_metadata_filter(raw: Optional[dict]) -> Optional[dict]:
    """Convert LLM-generated metadata_filter into valid Chroma where syntax.

    Handles two common LLM mistakes:
    - Wrong key names (official_sources, subreddit, date_range …)
    - Multiple top-level keys without a $and wrapper
    """
    if not raw:
        return None

    conditions: list[dict] = []
    for key, val in raw.items():
        # Remap known aliases
        canon = _KEY_ALIASES.get(key, key)
        # Drop community-side keys and anything not in the official schema
        if canon in _COMMUNITY_KEYS or canon not in _VALID_FILTER_KEYS:
            log.debug("hybrid.filter_key_dropped", key=key, canon=canon)
            continue
        if isinstance(val, list):
            if not val:
                continue
            if len(val) == 1:
                conditions.append({canon: {"$eq": val[0]}})
            else:
                conditions.append({canon: {"$in": val}})
        elif isinstance(val, str):
            conditions.append({canon: {"$eq": val}})
        # Non-string/list values (int, bool …) passed through as-is
        else:
            conditions.append({canon: {"$eq": val}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── BM25 helper (lazy import + LRU cache) ────────────────────────────────────

def _bm25_score_subset(
    documents: list[dict], query: str, top_k: int,
) -> list[tuple[int, float]]:
    """Return [(corpus_index, score)] sorted desc, length up to top_k.

    The BM25Okapi index is memoised by the corpus fingerprint
    (sha256 of the sorted chunk ids), so repeated queries against the
    same Chroma 1 candidate slice reuse the index. Cache invalidates
    when `services.bm25_cache.bump_corpus_version()` is called (the
    OfficialIngestionPipeline does this after upserting new chunks).

    Returns an empty list if rank_bm25 is unavailable.
    """
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:
        log.warning("hybrid.bm25_missing",
                    hint="pip install rank-bm25 to enable BM25 recall")
        return []
    if not documents:
        return []
    query_tokens = (query or "").lower().split()
    if not query_tokens:
        return []

    from services.bm25_cache import BM25_CACHE, fingerprint_corpus
    chunk_ids = [doc.get("id") or "" for doc in documents]
    fp = fingerprint_corpus(chunk_ids)
    bm25 = BM25_CACHE.get(fp)
    if bm25 is None:
        corpus = [
            (doc.get("document") or "").lower().split()
            for doc in documents
        ]
        bm25 = BM25Okapi(corpus)
        BM25_CACHE.put(fp, bm25)
    scores = bm25.get_scores(query_tokens)
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [(i, float(s)) for i, s in indexed[:top_k] if s > 0]


# ── Reranker (lazy import) ───────────────────────────────────────────────────

@dataclass
class _Reranker:
    model_name: str = "BAAI/bge-reranker-base"
    _model: object = None
    _available: bool = False

    def warmup(self) -> None:
        if self._available or self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self.model_name)
            self._available = True
        except Exception as exc:
            log.warning("hybrid.rerank_unavailable",
                        model=self.model_name,
                        error=str(exc)[:120])
            self._available = False

    def score(self, query: str, candidates: list[str]) -> list[float]:
        self.warmup()
        if not self._available or not candidates:
            return []
        pairs = [(query, c) for c in candidates]
        try:
            scores = self._model.predict(pairs)  # type: ignore[attr-defined]
            return [float(s) for s in scores]
        except Exception as exc:
            log.error("hybrid.rerank_error", error=str(exc)[:120])
            return []


# ── Hybrid Retrieval ────────────────────────────────────────────────────────

@dataclass
class HybridRetriever:
    collections: ChromaCollections = field(default_factory=ChromaCollections)
    embeddings: EmbeddingsService = field(default_factory=EmbeddingsService)
    reranker: _Reranker = field(default_factory=_Reranker)
    rrf_k: int = 60
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rerank_top_k: int = 20
    final_top_k: int = 10

    def __post_init__(self) -> None:
        self.reranker.warmup()

    def retrieve(
        self,
        query: str,
        *,
        metadata_filter: Optional[dict] = None,
        rerank: bool = True,
    ) -> EvidenceBundle:
        t0 = time.monotonic()
        metadata_filter = _normalize_metadata_filter(metadata_filter)
        bundle = EvidenceBundle(query=query, metadata_filter=metadata_filter or {})

        if not query or not query.strip():
            bundle.notes.append("empty_query")
            bundle.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return bundle

        # 1. Dense recall over Chroma 1 with metadata pre-filter
        embedding = self.embeddings.embed(query)
        dense_results = self.collections.official.query(
            embedding=embedding,
            n_results=self.dense_top_k,
            where=metadata_filter or None,
        )
        if not dense_results:
            bundle.notes.append("dense_recall_empty")

        dense_by_id = {r["id"]: r for r in dense_results}
        dense_ranks = {r["id"]: i + 1 for i, r in enumerate(dense_results)}

        # 2. BM25 recall over the same metadata-filtered subset
        # Reuse the dense results' documents to keep BM25 corpus aligned
        # with the pre-filter (avoids fetching the entire collection).
        bm25_pairs = _bm25_score_subset(dense_results, query, self.bm25_top_k)
        bm25_ranks = {dense_results[idx]["id"]: rank + 1
                      for rank, (idx, _score) in enumerate(bm25_pairs)}

        # 3. RRF fusion
        all_ids = set(dense_by_id) | set(bm25_ranks)
        fused: list[tuple[str, float]] = []
        for rid in all_ids:
            score = 0.0
            if rid in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[rid])
            if rid in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[rid])
            fused.append((rid, score))
        fused.sort(key=lambda x: x[1], reverse=True)

        # 4. Build initial chunks (top rerank_top_k)
        candidates: list[EvidenceChunk] = []
        for rid, rrf_score in fused[: self.rerank_top_k]:
            r = dense_by_id.get(rid)
            if r is None:
                continue
            meta = r.get("metadata") or {}
            chunk = EvidenceChunk(
                chunk_id=rid,
                text=r.get("document", ""),
                citation=Citation(
                    chunk_id=rid,
                    source=meta.get("source", "unknown"),
                    domain=meta.get("domain", ""),
                    tier=meta.get("tier", "reputable_media"),
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                ),
                dense_rank=dense_ranks.get(rid),
                bm25_rank=bm25_ranks.get(rid),
                rrf_score=rrf_score,
            )
            candidates.append(chunk)

        # 5. Rerank
        if rerank and candidates:
            rerank_scores = self.reranker.score(
                query, [c.text for c in candidates],
            )
            if rerank_scores:
                bundle.rerank_used = True
                for c, s in zip(candidates, rerank_scores):
                    c.rerank_score = s
                candidates.sort(
                    key=lambda c: (c.rerank_score if c.rerank_score is not None
                                   else c.rrf_score),
                    reverse=True,
                )
            else:
                bundle.notes.append("rerank_skipped")

        # 6. Final cut
        for i, c in enumerate(candidates[: self.final_top_k]):
            c.final_rank = i + 1
        bundle.chunks = candidates[: self.final_top_k]

        bundle.elapsed_ms = int((time.monotonic() - t0) * 1000)
        log.info("hybrid.retrieve_done",
                 query=query[:60],
                 dense=len(dense_results),
                 bm25=len(bm25_ranks),
                 candidates=len(candidates),
                 final=len(bundle.chunks),
                 rerank_used=bundle.rerank_used,
                 elapsed_ms=bundle.elapsed_ms)
        return bundle
