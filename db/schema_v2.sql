-- ============================================================
-- society_db v2 schema - redesign-2026-05 Phase 2
-- Run once: psql -d society_db -f db/schema_v2.sql
--
-- Coexists with the v1 schema. v2 introduces:
--   - posts_v2: fixed core columns + JSONB extra (Schema-aware Agent
--               proposes which fields go into extra; never ALTER TABLE)
--   - simhash bigint for post-level dedup (Hamming distance <= 3)
--   - tsvector + pg_trgm for full-text fallback search
--   - schema_meta: schema fingerprint + per-column descriptions
--                  (mirrored to Chroma 2; see scripts/rebuild_chroma2_schema.py)
--   - topics_v2: post-level cluster summaries
--   - entities_v2: extracted PERSON/ORG/LOC/EVENT entities
--   - reflection_log: audit trail for Critic verdicts (Phase 5 reads)
-- ============================================================

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------- posts_v2 -------------------------------------------------------
CREATE TABLE IF NOT EXISTS posts_v2 (
    post_id           TEXT PRIMARY KEY,
    account_id        TEXT NOT NULL,
    author            TEXT NOT NULL,        -- human-readable author handle
    text              TEXT NOT NULL,        -- compressed / merged text
    posted_at         TIMESTAMPTZ,
    subreddit         TEXT,                 -- nullable for non-Reddit sources
    source            TEXT NOT NULL DEFAULT 'reddit',  -- reddit | fixture
    topic_id          TEXT,                 -- assigned by post-level clusterer
    dominant_emotion  TEXT,                 -- fear | anger | hope | disgust | neutral
    emotion_score     REAL DEFAULT 0.0,
    -- Engagement metrics: stable across platforms, indexable, queryable.
    like_count        INTEGER NOT NULL DEFAULT 0,
    reply_count       INTEGER NOT NULL DEFAULT 0,
    retweet_count     INTEGER NOT NULL DEFAULT 0,
    simhash           BIGINT,               -- 64-bit simhash for dedup
    text_tsv          tsvector,             -- generated below
    extra             JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- production hardening Day 2-3: run lineage so we can answer
    -- "which pipeline run produced this row" and roll back failed runs.
    first_seen_in_run TEXT NOT NULL DEFAULT 'legacy_pre_v6',
    last_updated_in_run TEXT NOT NULL DEFAULT 'legacy_pre_v6'
);

-- Generated tsvector column maintained by trigger to avoid PG version
-- compatibility issues with GENERATED ALWAYS AS columns.
CREATE OR REPLACE FUNCTION posts_v2_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.text_tsv := to_tsvector('english', COALESCE(NEW.text, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS posts_v2_tsv_update ON posts_v2;
CREATE TRIGGER posts_v2_tsv_update
    BEFORE INSERT OR UPDATE OF text ON posts_v2
    FOR EACH ROW EXECUTE FUNCTION posts_v2_tsv_trigger();

CREATE INDEX IF NOT EXISTS idx_posts_v2_topic       ON posts_v2(topic_id);
CREATE INDEX IF NOT EXISTS idx_posts_v2_author      ON posts_v2(author);
CREATE INDEX IF NOT EXISTS idx_posts_v2_posted_at   ON posts_v2(posted_at DESC);
CREATE INDEX IF NOT EXISTS idx_posts_v2_simhash     ON posts_v2(simhash);
CREATE INDEX IF NOT EXISTS idx_posts_v2_emotion     ON posts_v2(dominant_emotion);
CREATE INDEX IF NOT EXISTS idx_posts_v2_like_count   ON posts_v2(like_count DESC);
CREATE INDEX IF NOT EXISTS idx_posts_v2_extra_gin   ON posts_v2 USING GIN (extra jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_posts_v2_text_tsv    ON posts_v2 USING GIN (text_tsv);
CREATE INDEX IF NOT EXISTS idx_posts_v2_text_trgm   ON posts_v2 USING GIN (text gin_trgm_ops);

-- ---------- topics_v2 ------------------------------------------------------
CREATE TABLE IF NOT EXISTS topics_v2 (
    topic_id          TEXT PRIMARY KEY,
    label             TEXT NOT NULL,
    post_count        INTEGER NOT NULL DEFAULT 0,
    dominant_emotion  TEXT,
    centroid_text     TEXT,         -- representative post text
    extra             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    first_seen_in_run TEXT NOT NULL DEFAULT 'legacy_pre_v6',
    last_updated_in_run TEXT NOT NULL DEFAULT 'legacy_pre_v6'
);

-- topics_v2.post_count is recomputed by trigger from posts_v2.topic_id so it
-- never drifts from reality. Keeps "hot topic" ranking honest after rollbacks
-- and post deletions.
CREATE OR REPLACE FUNCTION refresh_topic_post_count() RETURNS trigger AS $$
DECLARE
    affected TEXT[];
BEGIN
    IF TG_OP = 'DELETE' THEN
        affected := ARRAY[OLD.topic_id];
    ELSIF TG_OP = 'INSERT' THEN
        affected := ARRAY[NEW.topic_id];
    ELSE
        affected := ARRAY[NEW.topic_id, OLD.topic_id];
    END IF;

    UPDATE topics_v2 t
    SET post_count = (
            SELECT COUNT(*) FROM posts_v2 p WHERE p.topic_id = t.topic_id
        ),
        updated_at = NOW()
    WHERE t.topic_id = ANY(affected)
      AND t.topic_id IS NOT NULL;
    RETURN NULL;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS posts_v2_topic_count_ins ON posts_v2;
CREATE TRIGGER posts_v2_topic_count_ins
    AFTER INSERT ON posts_v2
    FOR EACH ROW EXECUTE FUNCTION refresh_topic_post_count();

DROP TRIGGER IF EXISTS posts_v2_topic_count_upd ON posts_v2;
CREATE TRIGGER posts_v2_topic_count_upd
    AFTER UPDATE OF topic_id ON posts_v2
    FOR EACH ROW
    WHEN (OLD.topic_id IS DISTINCT FROM NEW.topic_id)
    EXECUTE FUNCTION refresh_topic_post_count();

DROP TRIGGER IF EXISTS posts_v2_topic_count_del ON posts_v2;
CREATE TRIGGER posts_v2_topic_count_del
    AFTER DELETE ON posts_v2
    FOR EACH ROW EXECUTE FUNCTION refresh_topic_post_count();

-- ---------- entities_v2 ----------------------------------------------------
CREATE TABLE IF NOT EXISTS entities_v2 (
    entity_id         TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    entity_type       TEXT NOT NULL CHECK (entity_type IN
                          ('PERSON','ORG','LOC','EVENT','OTHER')),
    mention_count     INTEGER NOT NULL DEFAULT 1,
    extra             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    first_seen_in_run TEXT NOT NULL DEFAULT 'legacy_pre_v6',
    last_updated_in_run TEXT NOT NULL DEFAULT 'legacy_pre_v6'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_v2_name_type
    ON entities_v2 (LOWER(name), entity_type);

CREATE TABLE IF NOT EXISTS post_entities_v2 (
    post_id           TEXT NOT NULL REFERENCES posts_v2(post_id) ON DELETE CASCADE,
    entity_id         TEXT NOT NULL REFERENCES entities_v2(entity_id) ON DELETE CASCADE,
    char_start        INTEGER,
    char_end          INTEGER,
    confidence        REAL DEFAULT 0.5,
    PRIMARY KEY (post_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_post_entities_v2_entity
    ON post_entities_v2(entity_id);

-- ---------- claims_v2 ------------------------------------------------------
-- Atomic claims extracted per Reddit post (and from parent submission
-- titles for link-post replies). Stored as first-class so:
--   - "list claims in topic T" / "list claims about X" answer from PG, not
--     from LLM hallucination over post text;
--   - fact-check queries can match by embedding cosine + simhash hamming +
--     tsvector keyword without scraping post text at query time.
-- Many-to-many to posts_v2: same claim is often paraphrased by multiple
-- commenters; a single comment can also reference multiple claims.
CREATE TABLE IF NOT EXISTS claims_v2 (
    claim_id          TEXT PRIMARY KEY,        -- sha256(normalized_text)[:12]
    claim_text        TEXT NOT NULL,
    topic_id          TEXT,                    -- FK to topics_v2; nullable until clusterer assigns
    embedding         vector(1536),            -- pgvector; NULL if embed failed
    simhash           BIGINT,                  -- 64-bit simhash for near-dup search
    claim_text_tsv    tsvector,                -- maintained by trigger below
    source_url        TEXT,                    -- linked article URL (for link-post claims) or null
    role              TEXT,                    -- 'asserts' | 'discusses' | 'rebuts' | 'parent_context'
    confidence        REAL DEFAULT 0.5,        -- LLM confidence
    extra             JSONB NOT NULL DEFAULT '{}'::jsonb,
    first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extracted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    first_seen_in_run TEXT NOT NULL DEFAULT 'legacy_pre_v6',
    last_updated_in_run TEXT NOT NULL DEFAULT 'legacy_pre_v6'
);

CREATE OR REPLACE FUNCTION claims_v2_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.claim_text_tsv := to_tsvector('english', COALESCE(NEW.claim_text, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS claims_v2_tsv_update ON claims_v2;
CREATE TRIGGER claims_v2_tsv_update
    BEFORE INSERT OR UPDATE OF claim_text ON claims_v2
    FOR EACH ROW EXECUTE FUNCTION claims_v2_tsv_trigger();

CREATE INDEX IF NOT EXISTS idx_claims_v2_topic         ON claims_v2(topic_id);
CREATE INDEX IF NOT EXISTS idx_claims_v2_simhash       ON claims_v2(simhash);
CREATE INDEX IF NOT EXISTS idx_claims_v2_text_tsv      ON claims_v2 USING GIN (claim_text_tsv);
CREATE INDEX IF NOT EXISTS idx_claims_v2_text_trgm     ON claims_v2 USING GIN (claim_text gin_trgm_ops);
-- pgvector ANN index. Cosine distance; 16 connections, ef_construction 64.
-- Built lazily; ok if collection is initially small.
CREATE INDEX IF NOT EXISTS idx_claims_v2_embedding_hnsw
    ON claims_v2 USING hnsw (embedding vector_cosine_ops);

-- post_claims_v2: M:N link
CREATE TABLE IF NOT EXISTS post_claims_v2 (
    post_id   TEXT NOT NULL REFERENCES posts_v2(post_id) ON DELETE CASCADE,
    claim_id  TEXT NOT NULL REFERENCES claims_v2(claim_id) ON DELETE CASCADE,
    role      TEXT,                              -- per-link role override
    PRIMARY KEY (post_id, claim_id)
);
CREATE INDEX IF NOT EXISTS idx_post_claims_v2_claim ON post_claims_v2(claim_id);

-- ---------- schema_meta ----------------------------------------------------
-- Per-column descriptions and fingerprints. Mirrored to Chroma 2 by
-- agents/schema_agent.py (see Phase 2 5b double-write contract).
CREATE TABLE IF NOT EXISTS schema_meta (
    id                SERIAL PRIMARY KEY,
    table_name        TEXT NOT NULL,
    column_name       TEXT NOT NULL,
    column_type       TEXT NOT NULL,
    description       TEXT NOT NULL,
    sample_values     JSONB NOT NULL DEFAULT '[]'::jsonb,
    fingerprint       TEXT NOT NULL,     -- sha256(table.column.type)
    in_extra          BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (table_name, column_name)
);
CREATE INDEX IF NOT EXISTS idx_schema_meta_table ON schema_meta(table_name);

-- ---------- reflection_log -------------------------------------------------
-- Phase 5 audit table for Critic verdicts. Errors also routed to
-- Chroma 2 / Chroma 3 by services/reflection_store.py.
CREATE TABLE IF NOT EXISTS reflection_log (
    id                BIGSERIAL PRIMARY KEY,
    occurred_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id        TEXT,
    user_message      TEXT,
    error_kind        TEXT,
    failed_branch     TEXT,
    causal_record_ids TEXT[],
    payload           JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_reflection_kind ON reflection_log(error_kind);
CREATE INDEX IF NOT EXISTS idx_reflection_time ON reflection_log(occurred_at DESC);
