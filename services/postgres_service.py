"""
Postgres service — structured operational data store.
Wraps psycopg2; all queries use parameterised statements (no SQL injection).
"""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

import psycopg2
import psycopg2.extras
import structlog

from config import POSTGRES_DSN

log = structlog.get_logger(__name__)


class PostgresService:
    def __init__(self, dsn: str = POSTGRES_DSN) -> None:
        self._dsn = dsn
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        self._conn = psycopg2.connect(self._dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        self._conn.autocommit = False
        log.info("postgres.connected", dsn=self._dsn)

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    @contextmanager
    def cursor(self):
        # Attempt up to 2 tries: first with the existing connection, then after
        # reconnecting (handles cases where the server restarted under us).
        for attempt in range(2):
            try:
                if self._conn is None or self._conn.closed:
                    self.connect()
                cur = self._conn.cursor()
                try:
                    yield cur
                    self._conn.commit()
                except Exception:
                    try:
                        self._conn.rollback()
                    except Exception:
                        pass
                    raise
                finally:
                    cur.close()
                return  # success — exit the retry loop
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                if attempt == 0:
                    # Connection dropped (e.g. server restarted); reconnect once
                    log.warning("postgres.reconnecting", error=str(exc)[:80])
                    self._conn = None
                else:
                    raise

    # The v1 helpers (upsert_post / upsert_image / upsert_account /
    # upsert_claim / link_post_claim / increment_claim_propagation /
    # get_claim) were deleted in redesign-2026-05 Phase 5 along with the
    # `accounts / posts / images / claims / post_claims` tables. Use the
    # v2 helpers below (upsert_post_v2 / upsert_topic_v2 / ...).

    # ── v2 (redesign-2026-05) ────────────────────────────────────────────────

    def upsert_post_v2(
        self,
        *,
        post_id: str,
        account_id: str,
        author: str,
        text: str,
        posted_at: Optional[datetime] = None,
        subreddit: Optional[str] = None,
        source: str = "reddit",
        topic_id: Optional[str] = None,
        dominant_emotion: Optional[str] = None,
        emotion_score: float = 0.0,
        like_count: int = 0,
        reply_count: int = 0,
        retweet_count: int = 0,
        simhash: Optional[int] = None,
        extra: Optional[dict] = None,
        run_id: str = "legacy_pre_v6",
    ) -> None:
        """Upsert into posts_v2 (PROJECT_REDESIGN_V2.md Phase 2 5b).

        `run_id` writes lineage columns: first_seen_in_run is set on the
        initial INSERT, last_updated_in_run is bumped on every UPSERT.
        """
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO posts_v2 (post_id, account_id, author, text,
                    posted_at, subreddit, source, topic_id,
                    dominant_emotion, emotion_score,
                    like_count, reply_count, retweet_count,
                    simhash, extra,
                    first_seen_in_run, last_updated_in_run)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (post_id) DO UPDATE SET
                    text             = EXCLUDED.text,
                    topic_id         = EXCLUDED.topic_id,
                    dominant_emotion = EXCLUDED.dominant_emotion,
                    emotion_score    = EXCLUDED.emotion_score,
                    like_count       = EXCLUDED.like_count,
                    reply_count      = EXCLUDED.reply_count,
                    retweet_count    = EXCLUDED.retweet_count,
                    simhash          = EXCLUDED.simhash,
                    extra            = posts_v2.extra || EXCLUDED.extra,
                    last_updated_in_run = EXCLUDED.last_updated_in_run
            """, (
                post_id, account_id, author, text, posted_at,
                subreddit, source, topic_id, dominant_emotion,
                emotion_score,
                like_count, reply_count, retweet_count,
                simhash,
                json.dumps(extra or {}),
                run_id, run_id,
            ))

    def upsert_topic_v2(
        self,
        *,
        topic_id: str,
        label: str,
        post_count: int,
        dominant_emotion: Optional[str] = None,
        centroid_text: Optional[str] = None,
        extra: Optional[dict] = None,
        run_id: str = "legacy_pre_v6",
    ) -> None:
        # post_count is authoritative-maintained by the refresh_topic_post_count
        # trigger on posts_v2; pass it through here only as the initial value
        # for brand-new topic rows. The ON CONFLICT branch deliberately does
        # NOT overwrite post_count, so the trigger-maintained truth wins.
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO topics_v2 (topic_id, label, post_count,
                    dominant_emotion, centroid_text, extra, updated_at,
                    first_seen_in_run, last_updated_in_run)
                VALUES (%s,%s,%s,%s,%s,%s,NOW(),%s,%s)
                ON CONFLICT (topic_id) DO UPDATE SET
                    label            = EXCLUDED.label,
                    dominant_emotion = EXCLUDED.dominant_emotion,
                    centroid_text    = EXCLUDED.centroid_text,
                    extra            = topics_v2.extra || EXCLUDED.extra,
                    updated_at       = NOW(),
                    last_updated_in_run = EXCLUDED.last_updated_in_run
            """, (topic_id, label, post_count, dominant_emotion,
                  centroid_text, json.dumps(extra or {}),
                  run_id, run_id))

    def upsert_entity_v2(
        self,
        *,
        entity_id: str,
        name: str,
        entity_type: str,
        mention_count: int = 1,
        run_id: str = "legacy_pre_v6",
    ) -> None:
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO entities_v2 (entity_id, name, entity_type,
                    mention_count, first_seen_in_run, last_updated_in_run)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (entity_id) DO UPDATE SET
                    mention_count = entities_v2.mention_count + EXCLUDED.mention_count,
                    last_updated_in_run = EXCLUDED.last_updated_in_run
            """, (entity_id, name, entity_type, mention_count,
                  run_id, run_id))

    def link_post_entity_v2(
        self,
        *,
        post_id: str,
        entity_id: str,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        confidence: float = 0.5,
    ) -> None:
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO post_entities_v2 (post_id, entity_id,
                    char_start, char_end, confidence)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (post_id, entity_id) DO UPDATE SET
                    confidence = EXCLUDED.confidence
            """, (post_id, entity_id, char_start, char_end, confidence))

    # ── claims_v2 (atomic claims for fact-check / topic-claim-audit) ────────

    def upsert_claim_v2(
        self,
        *,
        claim_id: str,
        claim_text: str,
        topic_id: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        simhash: Optional[int] = None,
        source_url: Optional[str] = None,
        role: Optional[str] = None,
        confidence: float = 0.5,
        run_id: str = "legacy_pre_v6",
    ) -> None:
        """Upsert into claims_v2. embedding is a 1536-dim list[float]; pgvector
        accepts the bracketed-string serialisation."""
        emb_str = None
        if embedding is not None:
            emb_str = "[" + ",".join(f"{v:.7f}" for v in embedding) + "]"
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO claims_v2 (claim_id, claim_text, topic_id,
                    embedding, simhash, source_url, role, confidence,
                    first_seen_in_run, last_updated_in_run)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (claim_id) DO UPDATE SET
                    topic_id    = COALESCE(EXCLUDED.topic_id, claims_v2.topic_id),
                    embedding   = COALESCE(EXCLUDED.embedding, claims_v2.embedding),
                    simhash     = COALESCE(EXCLUDED.simhash, claims_v2.simhash),
                    source_url  = COALESCE(EXCLUDED.source_url, claims_v2.source_url),
                    confidence  = GREATEST(EXCLUDED.confidence, claims_v2.confidence),
                    extracted_at = NOW(),
                    last_updated_in_run = EXCLUDED.last_updated_in_run
            """, (claim_id, claim_text, topic_id, emb_str, simhash,
                  source_url, role, confidence, run_id, run_id))

    def link_post_claim_v2(
        self,
        *,
        post_id: str,
        claim_id: str,
        role: Optional[str] = None,
    ) -> None:
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO post_claims_v2 (post_id, claim_id, role)
                VALUES (%s,%s,%s)
                ON CONFLICT (post_id, claim_id) DO UPDATE SET
                    role = COALESCE(EXCLUDED.role, post_claims_v2.role)
            """, (post_id, claim_id, role))

    def search_claims_hybrid(
        self,
        query_text: str,
        query_embedding: Optional[list[float]],
        query_simhash: Optional[int],
        *,
        topic_id: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        """Three independent ranked lists for RRF fusion downstream.
        Returns {'cosine': [...], 'simhash': [...], 'tsv': [...]}."""
        emb_str = None
        if query_embedding is not None:
            emb_str = "[" + ",".join(f"{v:.7f}" for v in query_embedding) + "]"
        topic_clause = "AND topic_id = %s" if topic_id else ""
        params_topic = (topic_id,) if topic_id else ()

        cosine: list[dict] = []
        simhash_hits: list[dict] = []
        tsv_hits: list[dict] = []
        with self.cursor() as cur:
            if emb_str is not None:
                cur.execute(f"""
                    SELECT claim_id, claim_text, topic_id, source_url,
                           simhash, role, confidence,
                           1 - (embedding <=> %s::vector) AS score
                    FROM claims_v2
                    WHERE embedding IS NOT NULL
                      {topic_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (emb_str, *params_topic, emb_str, limit))
                cosine = list(cur.fetchall())
            if query_simhash is not None:
                # Postgres lacks a builtin popcount on bigint XOR efficiently
                # without an extension; do an exact pre-filter via topic and
                # fetch a wider set then filter in Python via hamming.
                # Cheap path: just pick rows with the same simhash first.
                cur.execute(f"""
                    SELECT claim_id, claim_text, topic_id, source_url,
                           simhash, role, confidence
                    FROM claims_v2
                    WHERE simhash IS NOT NULL
                      {topic_clause}
                    LIMIT 500
                """, params_topic)
                rows = list(cur.fetchall())
                # Compute hamming in Python and rank closest-first.
                def _hd(a: int, b: int) -> int:
                    mask = (1 << 64) - 1
                    return ((a & mask) ^ (b & mask)).bit_count()
                rows.sort(key=lambda r: _hd(int(r["simhash"]), int(query_simhash)))
                simhash_hits = rows[:limit]
            if query_text:
                cur.execute(f"""
                    SELECT claim_id, claim_text, topic_id, source_url,
                           simhash, role, confidence,
                           ts_rank(claim_text_tsv,
                                   plainto_tsquery('english', %s)) AS score
                    FROM claims_v2
                    WHERE claim_text_tsv @@ plainto_tsquery('english', %s)
                      {topic_clause}
                    ORDER BY score DESC
                    LIMIT %s
                """, (query_text, query_text, *params_topic, limit))
                tsv_hits = list(cur.fetchall())
        return {"cosine": cosine, "simhash": simhash_hits, "tsv": tsv_hits}

    def list_claims_for_topic(
        self, topic_id: str, *, limit: int = 50,
    ) -> list[dict]:
        with self.cursor() as cur:
            cur.execute("""
                SELECT c.claim_id, c.claim_text, c.source_url, c.role,
                       c.confidence, c.first_seen_at,
                       COUNT(pc.post_id) AS post_count
                FROM claims_v2 c
                LEFT JOIN post_claims_v2 pc ON pc.claim_id = c.claim_id
                WHERE c.topic_id = %s
                GROUP BY c.claim_id
                ORDER BY post_count DESC, c.first_seen_at DESC
                LIMIT %s
            """, (topic_id, limit))
            return list(cur.fetchall())

    # ── Run lineage (production hardening Day 4) ─────────────────────────────

    def delete_run_data(self, run_id: str) -> dict:
        """Hard-delete every row whose first_seen_in_run == run_id.

        Used by the rollback scanner when a pipeline left commit_state in
        'pending' or 'failed'. Safe on an empty / non-existent run.
        Returns counts per table for the audit log.
        """
        deleted = {"post_entities_v2": 0, "posts_v2": 0,
                   "topics_v2": 0, "entities_v2": 0,
                   "claims_v2": 0}
        with self.cursor() as cur:
            # Order matters: kill child FK rows first.
            cur.execute(
                "DELETE FROM post_entities_v2 "
                "WHERE post_id IN ("
                "  SELECT post_id FROM posts_v2 WHERE first_seen_in_run = %s"
                ")",
                (run_id,),
            )
            deleted["post_entities_v2"] = cur.rowcount
            # post_claims_v2 cascades via posts_v2 FK.
            cur.execute(
                "DELETE FROM posts_v2 WHERE first_seen_in_run = %s",
                (run_id,),
            )
            deleted["posts_v2"] = cur.rowcount
            cur.execute(
                "DELETE FROM topics_v2 WHERE first_seen_in_run = %s",
                (run_id,),
            )
            deleted["topics_v2"] = cur.rowcount
            cur.execute(
                "DELETE FROM entities_v2 WHERE first_seen_in_run = %s",
                (run_id,),
            )
            deleted["entities_v2"] = cur.rowcount
            cur.execute(
                "DELETE FROM claims_v2 WHERE first_seen_in_run = %s",
                (run_id,),
            )
            deleted["claims_v2"] = cur.rowcount
        return deleted

    # ── schema_meta (Schema-aware Agent double-write) ────────────────────────

    def upsert_schema_meta(
        self,
        *,
        table_name: str,
        column_name: str,
        column_type: str,
        description: str,
        sample_values: list[str],
        fingerprint: str,
        in_extra: bool,
    ) -> None:
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO schema_meta (table_name, column_name, column_type,
                    description, sample_values, fingerprint, in_extra, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
                ON CONFLICT (table_name, column_name) DO UPDATE SET
                    column_type   = EXCLUDED.column_type,
                    description   = EXCLUDED.description,
                    sample_values = EXCLUDED.sample_values,
                    fingerprint   = EXCLUDED.fingerprint,
                    in_extra      = EXCLUDED.in_extra,
                    updated_at    = NOW()
            """, (table_name, column_name, column_type, description,
                  json.dumps(sample_values), fingerprint, in_extra))

    def list_schema_meta(self, table_name: Optional[str] = None) -> list[dict]:
        with self.cursor() as cur:
            if table_name:
                cur.execute(
                    "SELECT * FROM schema_meta WHERE table_name = %s "
                    "ORDER BY column_name", (table_name,),
                )
            else:
                cur.execute(
                    "SELECT * FROM schema_meta ORDER BY table_name, column_name"
                )
            return list(cur.fetchall())

    def list_information_schema_columns(
        self, table_name: str = "posts_v2",
    ) -> list[dict]:
        """Pull live column list from PG (used by consistency tests + rebuild)."""
        with self.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table_name,))
            return list(cur.fetchall())

    # ── Full-text + similarity search (Posts not vectorised) ─────────────────

    def search_posts_fulltext(
        self, query: str, limit: int = 50,
    ) -> list[dict]:
        """English tsvector search; powers the NL2SQL keyword fallback."""
        with self.cursor() as cur:
            cur.execute("""
                SELECT post_id, author, text, posted_at, topic_id,
                       ts_rank(text_tsv, plainto_tsquery('english', %s)) AS rank
                FROM posts_v2
                WHERE text_tsv @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC LIMIT %s
            """, (query, query, limit))
            return list(cur.fetchall())

    def search_posts_trgm(
        self, query: str, similarity_threshold: float = 0.3, limit: int = 20,
    ) -> list[dict]:
        """pg_trgm fuzzy match, used as long-text dedup fallback."""
        with self.cursor() as cur:
            cur.execute("""
                SELECT post_id, author, text, posted_at, topic_id,
                       similarity(text, %s) AS sim
                FROM posts_v2
                WHERE similarity(text, %s) >= %s
                ORDER BY sim DESC LIMIT %s
            """, (query, query, similarity_threshold, limit))
            return list(cur.fetchall())

    def find_simhash_neighbours(
        self, simhash: int, limit: int = 32,
    ) -> list[dict]:
        """Return candidate near-duplicate posts by exact simhash equality.

        Hamming-distance comparison against the candidates is done in Python by
        the dedup module (PG cannot easily compute population_count of XOR on
        signed BIGINT pre-PG14 across all distros).
        """
        with self.cursor() as cur:
            cur.execute("""
                SELECT post_id, simhash, text, ingested_at
                FROM posts_v2
                WHERE simhash IS NOT NULL
                ORDER BY ingested_at DESC LIMIT %s
            """, (limit,))
            return list(cur.fetchall())

    # v1's save_report() was deleted in redesign-2026-05 Phase 5: the
    # IncidentReport model is gone and the v2 chat path persists answers
    # via session_store + reflection_log.
