"""
seed_claims_nl2sql_examples.py

Seed Chroma 2 with:
  - schema docs for claims_v2 / post_claims_v2 (so NL2SQL knows the columns);
  - guides that steer fact-check / topic-claim-audit queries to claim_search
    instead of `posts_v2.text_tsv`;
  - error_lessons recording the historical failure mode where NL2SQL invented
    `posts_v2.source IN ('AP','BBC',...)` filters;
  - success exemplars demonstrating the right pattern (call claim_search, or
    SELECT FROM claims_v2 directly with embedding cosine).

Idempotent: schema docs use deterministic ids; guides too. success/error
exemplars dedup via conflict policy.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import structlog

from services.embeddings_service import EmbeddingsService
from services.nl2sql_memory import NL2SQLMemory

log = structlog.get_logger(__name__)


# ── claims_v2 / post_claims_v2 schema docs ────────────────────────────────

@dataclass
class _SchemaSeed:
    table: str
    column: str
    text: str


SCHEMA_SEEDS: list[_SchemaSeed] = [
    _SchemaSeed("claims_v2", "claim_id",
        "claims_v2.claim_id (TEXT, primary key). Stable sha256 hash of the "
        "normalized claim text. Use to JOIN with post_claims_v2.claim_id."),
    _SchemaSeed("claims_v2", "claim_text",
        "claims_v2.claim_text (TEXT). Atomic factual claim extracted from a "
        "Reddit submission title. NOT raw post body. Use this column when the "
        "user asks 'list claims about X', NOT posts_v2.text."),
    _SchemaSeed("claims_v2", "topic_id",
        "claims_v2.topic_id (TEXT). Same topic_id space as topics_v2 / "
        "posts_v2. Filter by topic_id when scoping claims to one cluster."),
    _SchemaSeed("claims_v2", "embedding",
        "claims_v2.embedding (vector(1536), pgvector). Cosine distance via "
        "`embedding <=> '[...]'::vector`. Built for semantic claim search; "
        "do NOT compare with raw float[] casts."),
    _SchemaSeed("claims_v2", "simhash",
        "claims_v2.simhash (BIGINT). 64-bit Charikar simhash over claim "
        "tokens. Near-duplicates have hamming distance <= 3 via XOR + "
        "popcount. For end-user fact-check queries prefer the claim_search "
        "tool, which fuses simhash with cosine + tsvector."),
    _SchemaSeed("claims_v2", "source_url",
        "claims_v2.source_url (TEXT, nullable). Link to the article or page "
        "the claim originated from (extracted from the parent submission). "
        "Cite this in fact-check answers."),
    _SchemaSeed("claims_v2", "role",
        "claims_v2.role (TEXT). 'asserts' | 'discusses' | 'rebuts' | "
        "'parent_context'. Per-row role on the source post; per-link role "
        "lives in post_claims_v2.role."),
    _SchemaSeed("claims_v2", "claim_text_tsv",
        "claims_v2.claim_text_tsv (tsvector). Maintained by trigger over "
        "claim_text. Use `@@ plainto_tsquery('english', $1)` for keyword "
        "fallback when an exact phrase needs to match."),
    _SchemaSeed("claims_v2", "confidence",
        "claims_v2.confidence (REAL). Extractor confidence in [0,1]; default "
        "0.5. Filter low-confidence claims with `confidence >= 0.5`."),

    _SchemaSeed("post_claims_v2", "post_id",
        "post_claims_v2.post_id (TEXT). FK to posts_v2.post_id. JOIN here to "
        "get author / posted_at / subreddit for a claim."),
    _SchemaSeed("post_claims_v2", "claim_id",
        "post_claims_v2.claim_id (TEXT). FK to claims_v2.claim_id. M:N: one "
        "claim has many posts (paraphrases / reply chains); one post may "
        "discuss multiple claims."),
    _SchemaSeed("post_claims_v2", "role",
        "post_claims_v2.role (TEXT). 'asserts' | 'discusses' | 'rebuts' | "
        "'parent_context'. Use this to filter, e.g., authors who ASSERT a "
        "claim vs commenters who only discuss it."),
]


# ── Durable guidance (kind=guide) ───────────────────────────────────────────

@dataclass
class _GuideSeed:
    rule_id: str
    text: str
    category: str = "rule"
    priority: int = 70


GUIDE_SEEDS: list[_GuideSeed] = [
    _GuideSeed(
        rule_id="claims_use_claim_search_for_factcheck",
        text=(
            "FACT-CHECK / TOPIC-CLAIM-AUDIT QUERIES (hard rule):\n"
            "When the user asks to (a) verify / fact-check a Reddit claim, "
            "(b) list claims in a topic, or (c) classify Reddit claims as "
            "consistent / contradicted / lacking evidence vs official sources:\n"
            "  1. Read claims FROM claims_v2 (atomic, deduplicated) — NEVER "
            "     try to extract claim text from posts_v2.text. Comments "
            "     don't store the parent submission's title.\n"
            "  2. For semantic claim matching ('claim about X', 'claims like Y'), "
            "     emit a CLAIM_SEARCH tool call (preferred) OR write a "
            "     pgvector cosine query: "
            "     `claim_text, embedding <=> '[..query embedding..]'::vector AS dist "
            "     FROM claims_v2 WHERE topic_id=... ORDER BY dist LIMIT N`.\n"
            "  3. Use claims_v2.claim_text_tsv only for exact-phrase fallback.\n"
            "  4. Join claims_v2 -> post_claims_v2 -> posts_v2 to attribute "
            "     a claim to authors/posts. Cite claims_v2.source_url as the "
            "     external article link.\n"
        ),
        priority=90,
    ),
    _GuideSeed(
        rule_id="posts_v2_source_is_always_reddit",
        text=(
            "POSTS_V2.SOURCE IS A PLATFORM TAG:\n"
            "posts_v2.source is the social platform. Today the only value in "
            "production is 'reddit'. NEVER write `WHERE source IN ('AP', "
            "'BBC', 'Reuters', 'NYT', 'Xinhua', ...)` — those are official-"
            "news outlets, which live in chroma_official (Evidence Retrieval "
            "branch), NOT in posts_v2. A query that filters posts_v2.source "
            "by news outlet name is guaranteed to return zero rows.\n"
            "If the user mentions 'official sources / AP / Reuters / BBC / "
            "NYT', the corresponding evidence retrieval is handled by a "
            "different branch — your job in NL2SQL is the COMMUNITY side. "
            "Drop those source names from the WHERE clause; only filter on "
            "subreddit, topic_id, posted_at, dominant_emotion, etc."
        ),
        priority=95,
    ),
    _GuideSeed(
        rule_id="dont_search_posts_text_for_claim_phrasing",
        text=(
            "DO NOT keyword-match claim phrases against posts_v2.text:\n"
            "Reddit comments rarely repeat the parent submission's title "
            "verbatim. A query like "
            "`posts_v2.text_tsv @@ plainto_tsquery('Russia Victory Day "
            "parade...')` will return ZERO rows even when a claim about that "
            "topic is well-discussed. Match against claims_v2 instead "
            "(embedding cosine + simhash hamming + claim_text_tsv)."
        ),
        priority=85,
    ),
]


# ── Success exemplars ──────────────────────────────────────────────────────

@dataclass
class _SuccessSeed:
    nl: str
    sql: str
    table_hints: list[str]


SUCCESS_SEEDS: list[_SuccessSeed] = [
    _SuccessSeed(
        nl="List the top claims discussed in topic <TOPIC_ID> with their author count and source URL.",
        sql=(
            "SELECT c.claim_id, c.claim_text, c.source_url,\n"
            "       COUNT(DISTINCT p.author) AS distinct_authors,\n"
            "       COUNT(pc.post_id) AS post_count\n"
            "FROM claims_v2 c\n"
            "JOIN post_claims_v2 pc ON pc.claim_id = c.claim_id\n"
            "JOIN posts_v2 p ON p.post_id = pc.post_id\n"
            "WHERE c.topic_id = '<TOPIC_ID>'\n"
            "GROUP BY c.claim_id, c.claim_text, c.source_url\n"
            "ORDER BY post_count DESC\n"
            "LIMIT 20"
        ),
        table_hints=["claims_v2", "post_claims_v2", "posts_v2"],
    ),
    _SuccessSeed(
        nl="Find Reddit claims semantically similar to '<USER_CLAIM>' (fact-check candidate).",
        sql=(
            "-- Semantic claim match via pgvector cosine. Replace the literal "
            "vector with the embedding of the user's claim.\n"
            "SELECT c.claim_id, c.claim_text, c.topic_id, c.source_url,\n"
            "       1 - (c.embedding <=> '<QUERY_EMBEDDING>'::vector) AS similarity\n"
            "FROM claims_v2 c\n"
            "WHERE c.embedding IS NOT NULL\n"
            "ORDER BY c.embedding <=> '<QUERY_EMBEDDING>'::vector\n"
            "LIMIT 10"
        ),
        table_hints=["claims_v2"],
    ),
    _SuccessSeed(
        nl="Show me posts that assert the claim '<CLAIM_TEXT>' and who wrote them.",
        sql=(
            "WITH q AS (\n"
            "  SELECT claim_id FROM claims_v2\n"
            "  WHERE claim_text_tsv @@ plainto_tsquery('english', '<CLAIM_TEXT>')\n"
            "  ORDER BY ts_rank(claim_text_tsv,\n"
            "                   plainto_tsquery('english', '<CLAIM_TEXT>')) DESC\n"
            "  LIMIT 5\n"
            ")\n"
            "SELECT p.author, p.posted_at, p.subreddit, pc.role, c.claim_text\n"
            "FROM q\n"
            "JOIN claims_v2 c       USING (claim_id)\n"
            "JOIN post_claims_v2 pc USING (claim_id)\n"
            "JOIN posts_v2 p        USING (post_id)\n"
            "ORDER BY p.posted_at DESC\n"
            "LIMIT 50"
        ),
        table_hints=["claims_v2", "post_claims_v2", "posts_v2"],
    ),
    _SuccessSeed(
        nl="How many distinct claims has each topic surfaced, and which topic has the most distinct claims?",
        sql=(
            "SELECT t.topic_id, t.label,\n"
            "       COUNT(DISTINCT c.claim_id) AS distinct_claims\n"
            "FROM topics_v2 t\n"
            "LEFT JOIN claims_v2 c ON c.topic_id = t.topic_id\n"
            "GROUP BY t.topic_id, t.label\n"
            "ORDER BY distinct_claims DESC\n"
            "LIMIT 20"
        ),
        table_hints=["topics_v2", "claims_v2"],
    ),
]


# ── Error lessons (kind=error) ──────────────────────────────────────────────

@dataclass
class _ErrorSeed:
    failure_reason: str
    bad_pattern: str
    table_hints: list[str]


ERROR_SEEDS: list[_ErrorSeed] = [
    _ErrorSeed(
        failure_reason=(
            "posts_v2.source is the social platform tag (always 'reddit' "
            "today). Filtering posts_v2.source by news outlet names returns "
            "zero rows: official news lives in chroma_official, not posts_v2."
        ),
        bad_pattern=(
            "SELECT p.text, p.author, p.posted_at, p.source FROM posts_v2 p "
            "WHERE p.source IN ('AP', 'Reuters', 'BBC', 'NYT') "
            "AND p.text_tsv @@ plainto_tsquery('english', '<claim>')"
        ),
        table_hints=["posts_v2"],
    ),
    _ErrorSeed(
        failure_reason=(
            "Reddit comments rarely repeat the parent submission's title "
            "verbatim. Matching a claim phrase against posts_v2.text_tsv "
            "returns no rows even when the topic is heavily discussed. "
            "Search claims_v2 instead (embedding/simhash/tsv hybrid)."
        ),
        bad_pattern=(
            "SELECT * FROM posts_v2 WHERE text_tsv @@ "
            "plainto_tsquery('english', '<full claim sentence>')"
        ),
        table_hints=["posts_v2"],
    ),
]


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    embeddings = EmbeddingsService()
    memory = NL2SQLMemory()

    for sch in SCHEMA_SEEDS:
        emb = embeddings.embed(sch.text)
        rid = memory.upsert_schema(
            table_name=sch.table, column_name=sch.column,
            text=sch.text, embedding=emb,
            fingerprint=f"seed::{sch.table}.{sch.column}",
        )
        print(f"schema:{sch.table}.{sch.column} -> {rid}")

    for guide in GUIDE_SEEDS:
        emb = embeddings.embed(guide.text)
        rid = memory.upsert_guidance(
            rule_id=guide.rule_id, text=guide.text, embedding=emb,
            category=guide.category, priority=guide.priority,
        )
        print(f"guide:{guide.rule_id} -> {rid}")

    for seed in SUCCESS_SEEDS:
        text_for_embedding = f"NL: {seed.nl}\nSQL: {seed.sql}"
        emb = embeddings.embed(text_for_embedding)
        rid = memory.upsert_success(
            nl_query=seed.nl, sql_query=seed.sql, embedding=emb,
            table_hints=seed.table_hints,
        )
        print(f"success:{seed.nl[:60]}... -> {rid}")

    for err in ERROR_SEEDS:
        text_for_embedding = f"Avoid: {err.bad_pattern}\nReason: {err.failure_reason}"
        emb = embeddings.embed(text_for_embedding)
        rid = memory.upsert_error(
            failure_reason=err.failure_reason,
            bad_pattern=err.bad_pattern,
            embedding=emb,
            table_hints=err.table_hints,
        )
        print(f"error:{err.failure_reason[:60]}... -> {rid}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
