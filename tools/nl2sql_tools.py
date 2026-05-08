"""
NL2SQL tools - redesign-2026-05 Phase 3.3.

Branch B (NL2SQL). Three-step loop with internal repair (PROJECT_REDESIGN_V2.md
7b-(3) layered error handling):

    generate_sql()         - LLM produces a SELECT, fed by:
                                * Chroma 2 schema docs (kind=schema)
                                * Chroma 2 success exemplars (kind=success)
                                * Chroma 2 error lessons (kind=error)
    execute_and_validate() - Read-only DSN, SELECT-only whitelist,
                             LIMIT enforcement, statement_timeout.
    repair_sql()           - On error or unexpected empty rows, retry up
                             to NL2SQL_MAX_REPAIR_ROUNDS with the failure
                             folded into the prompt.

Errors that ARE Critic's concern (sql_empty_result with the wrong intent
match) bubble up via SQLOutput.success=False. Errors handled internally
(syntax / unknown column / timeout) only go to the repair loop and never
to Critic.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import openai
import psycopg2
import psycopg2.extras
import structlog

from config import (
    NL2SQL_MAX_REPAIR_ROUNDS,
    NL2SQL_RESULT_ROW_LIMIT,
    NL2SQL_STATEMENT_TIMEOUT_MS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    POSTGRES_DSN,
    POSTGRES_READONLY_DSN,
)
from models.branch_output import SQLAttempt, SQLOutput
from services.embeddings_service import EmbeddingsService
from services.nl2sql_memory import NL2SQLMemory

log = structlog.get_logger(__name__)


# ── Whitelist + safety helpers ───────────────────────────────────────────────

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|"
    r"COPY|VACUUM|ANALYZE|REINDEX|CLUSTER|REFRESH|EXECUTE|CALL|DO)\b",
    re.IGNORECASE,
)
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)


def _sanitise_sql(sql: str, row_limit: int) -> tuple[str, Optional[str]]:
    """Return (safe_sql, error_kind). Reject anything that isn't pure SELECT."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        return "", "sql_syntax"
    if ";" in text:
        return "", "sql_syntax"  # disallow multi-statement
    if not re.match(r"^\s*(WITH|SELECT)\b", text, re.IGNORECASE):
        return "", "sql_syntax"
    if _FORBIDDEN_KEYWORDS.search(text):
        return "", "sql_syntax"
    # Force a LIMIT
    m = _LIMIT_RE.search(text)
    if m:
        try:
            current = int(m.group(1))
            if current > row_limit:
                text = _LIMIT_RE.sub(f"LIMIT {row_limit}", text, count=1)
        except ValueError:
            return "", "sql_syntax"
    else:
        text = f"{text} LIMIT {row_limit}"
    return text, None


# ── Generation prompt ────────────────────────────────────────────────────────

_GENERATE_SYSTEM = """You write SAFE Postgres SELECT queries.
You will be given:
  - the user question
  - relevant column descriptions (schema hints)
  - a few correct (NL, SQL) examples
  - error lessons to avoid

Rules:
- Output ONLY one SELECT query (or a CTE WITH ... SELECT). No semicolons.
- CRITICAL: You MUST always produce a non-empty SQL query. If you are
  uncertain which exact pattern to use, produce your best attempt rather
  than leaving the sql field empty or null. An imperfect query that
  executes is always better than no query at all.
- The "Avoid these mistakes" section describes previously failed SQL
  PATTERNS (wrong table names, bad syntax, etc.). It does NOT mean you
  should avoid writing SQL for that topic — only avoid the specific
  anti-pattern shown. NL2SQL guidance takes priority when it conflicts
  with an error lesson.
- Treat "NL2SQL guidance from Chroma 2" as durable operating guidance.
  Follow it unless it conflicts with read-only SQL safety rules.
- Use only the tables and columns explicitly listed in schema hints OR
  explicitly named in the NL2SQL guidance from Chroma 2. Guidance-named
  columns (e.g. dominant_emotion, text_tsv) are always valid even when
  absent from the current schema hints list.
- For free-text matching on posts.text use the existing tsvector via
  `text_tsv @@ plainto_tsquery('english', '...')`. Avoid LIKE on huge text.
- Always add a LIMIT (the runtime will cap it).
- Prefer joining via posts_v2.topic_id when the question is topic-scoped.
- NEVER return raw machine IDs (topic_id, entity_id, post_id, account_id)
  as the only user-facing column. Always JOIN to a human-readable name:
    * topic_id    -> JOIN topics_v2 ON ... SELECT topics_v2.label
    * entity_id   -> JOIN entities_v2 ON ... SELECT entities_v2.name
    * account_id  -> SELECT posts_v2.author (already human-readable)
  IDs are fine alongside the readable name (e.g. SELECT label, topic_id),
  but never alone unless the user explicitly asked "what is the id of ...".
- "What topics are trending" -> SELECT label, post_count FROM topics_v2
  ORDER BY post_count DESC.
- When the user asks for post counts within a specific time/source/subreddit
  window, count matching rows from posts_v2 (COUNT(*)) after applying those
  filters. Do not use topics_v2.post_count unless the user asks for all-time
  topic totals.
- When the user asks for dominant emotion within a specific
  time/source/subreddit window, compute it from matching posts_v2 rows with
  mode() WITHIN GROUP (ORDER BY posts_v2.dominant_emotion), but fall back to
  topics_v2.dominant_emotion when all matching post-level emotions are NULL
  or empty. Use:
      COALESCE(
        mode() WITHIN GROUP (ORDER BY NULLIF(p.dominant_emotion, '')),
        MAX(NULLIF(t.dominant_emotion, '')),
        'Not specified'
      )
  The topic-level fallback is wrapped in MAX(...) so the surrounding query
  stays valid under PostgreSQL's GROUP BY rules - both branches of COALESCE
  are aggregates, which holds even when the query has no GROUP BY at all or
  groups only on a non-PK column. This avoids hiding a topic-level emotion
  just because individual post rows were not annotated.

Freshness rules (production hardening):
- DEFAULT time window when the user input mentions "trend",
  "trending", "propagation", "amplify", "spread", "viral", "today",
  "this week", or similar present-tense framing:
      WHERE posts_v2.posted_at >= NOW() - INTERVAL '30 days'
  Apply this even if not explicitly asked - users expect "recent".
- DO NOT apply the default window for fact-check / official-source
  recap / historical questions ("what did X say in March", "all-time
  most influential", "since the recall").
- If the user explicitly says "last week" / "in the past 7 days" /
  "last month", honour their range exactly.
- If the user says "all time" / "since the beginning", omit the WHERE
  clause entirely.

Topic filtering rules (read carefully -- this is where most NL2SQL errors come from):
- If the user prompt provides "Pre-resolved topic_ids" and the question is
  about one named topic, USE THEM as the filter and IGNORE label matching:
      WHERE posts_v2.topic_id IN ('topic_xxx', 'topic_yyy')
  These ids were chosen by an embedding-based semantic match, so they
  already cover the user's topic phrase even if the label text differs.
- If the user asks to list all topics / today's topics / topics from a
  subreddit or corpus, DO NOT narrow the query to a single pre-resolved
  topic_id. Return all matching topics for the requested time/source window.
- Otherwise, when the user names a topic by exact label, JOIN on topic_id:
      SELECT posts_v2.text, posts_v2.author, posts_v2.dominant_emotion
      FROM posts_v2
      JOIN topics_v2 ON posts_v2.topic_id = topics_v2.topic_id
      WHERE topics_v2.label = 'X'
- Fuzzy fallback (only when no pre-resolved ids and no exact label):
      WHERE LOWER(topics_v2.label) LIKE LOWER('%X%')
- text_tsv is for matching POST CONTENT keywords (e.g. "posts mentioning
  vaccines"), NOT for matching topic labels. Topic labels are short
  curated phrases that virtually never appear verbatim in post text.
- When the user asks "what was discussed", project the actual post
  content (posts_v2.text and friends), not just the topic label that
  they already gave you.

Respond as JSON: {"sql": "..."}.
"""


_BUILTIN_GUIDANCE: list[tuple[str, str, str, int]] = [
    (
        "db_intro_core_tables",
        "Database guide: posts_v2 stores community posts and comments. "
        "Use posts_v2.posted_at for time windows, posts_v2.subreddit for "
        "subreddit filters, and posts_v2.source only for platform filters "
        "(the social platform is reddit), "
        "posts_v2.topic_id to join topics_v2, and posts_v2.author for the "
        "human-readable account. topics_v2 stores topic_id, label, "
        "post_count, dominant_emotion, and centroid_text. entities_v2 stores "
        "entity_id, name, entity_type. Join through topic_id/entity_id rather "
        "than guessing labels from post text.",
        "database",
        100,
    ),
    (
        "reddit_source_vs_subreddit",
        "Rule: source and subreddit are different. For a question about "
        "worldnews/conspiracy/politics/etc., filter posts_v2.subreddit = "
        "'worldnews' (or the named subreddit). Do not write "
        "posts_v2.source = 'worldnews'; source should be the platform "
        "value 'reddit'.",
        "rule",
        100,
    ),
    (
        "broad_topic_listing_no_topic_hint",
        "Rule: for questions like 'list the topics', 'all topics', "
        "'today's worldnews topics', or requests for topic_id, label, post "
        "count, emotion, do not filter to a single pre-resolved topic_id. "
        "Use EXISTS or a join to posts_v2 only to enforce the requested "
        "time/source window. Return topics_v2.topic_id and topics_v2.label.",
        "rule",
        100,
    ),
    (
        "filtered_topic_post_count_from_posts",
        "Rule: if the user asks for post count while also specifying a "
        "time window, subreddit, or source corpus (for example today's "
        "worldnews data), the post count must be COUNT(*) over matching "
        "posts_v2 rows after those filters. Do not return topics_v2.post_count "
        "for that column; topics_v2.post_count is the broader topic total.",
        "rule",
        100,
    ),
    (
        "filtered_topic_dominant_emotion_from_posts",
        "Rule: if the user asks for dominant emotion while also specifying "
        "a time window, subreddit, or source corpus, compute dominant_emotion "
        "from the matching posts_v2 rows after those filters, but fall back "
        "to topics_v2.dominant_emotion when every matching post-level emotion "
        "is NULL or empty. Use COALESCE(mode() WITHIN GROUP (ORDER BY "
        "NULLIF(p.dominant_emotion, '')), MAX(NULLIF(t.dominant_emotion, '')), "
        "'Not specified') AS dominant_emotion. The MAX(...) wrapper on the "
        "topic-level fallback is required so PostgreSQL accepts the query "
        "without listing t.dominant_emotion in GROUP BY. Do not lose the "
        "topic-level emotion when post rows are unannotated.",
        "rule",
        100,
    ),
    (
        "topic_id_requested_preserve_id",
        "Rule: when the user explicitly asks for topic_id, include the raw "
        "topics_v2.topic_id in the SELECT list and include topics_v2.label "
        "alongside it. Do not replace topic_id with the label in SQL output.",
        "rule",
        95,
    ),
    (
        "topic_specific_queries_use_resolved_ids",
        "Rule: when the question is about one specific named topic and "
        "Pre-resolved topic_ids are supplied, filter posts_v2.topic_id or "
        "topics_v2.topic_id with IN (...). Do not use LIKE on post text to "
        "match topic labels.",
        "rule",
        90,
    ),
    (
        "today_worldnews_window",
        "Example: List the topics from today's worldnews data with topic_id, "
        "label, post count, and dominant emotion. SQL shape: SELECT "
        "t.topic_id, t.label, COUNT(*) AS post_count, "
        "COALESCE(mode() WITHIN GROUP (ORDER BY NULLIF(p.dominant_emotion, "
        "'')), MAX(NULLIF(t.dominant_emotion, '')), 'Not specified') AS "
        "dominant_emotion "
        "FROM posts_v2 p JOIN topics_v2 t ON p.topic_id = t.topic_id "
        "WHERE p.posted_at >= NOW() - INTERVAL '1 day' "
        "AND p.subreddit = 'worldnews' GROUP BY t.topic_id, t.label "
        "ORDER BY post_count DESC LIMIT 100",
        "example",
        90,
    ),
    (
        "single_topic_dominant_emotion",
        "Example: What is the dominant emotion for the Russia-Ukraine Conflict "
        "topic? SQL shape: SELECT "
        "COALESCE(mode() WITHIN GROUP (ORDER BY NULLIF(p.dominant_emotion, "
        "'')), MAX(NULLIF(t.dominant_emotion, '')), 'Not specified') AS "
        "dominant_emotion "
        "FROM posts_v2 p JOIN topics_v2 t ON p.topic_id = t.topic_id "
        "WHERE t.label ILIKE '%Russia%Ukraine%' LIMIT 1. "
        "Note: posts_v2.dominant_emotion and topics_v2.dominant_emotion are "
        "valid columns even when not shown in schema hints.",
        "example",
        90,
    ),
]

_BUILTIN_GUIDANCE_SEEDED = False
_LIVE_SCHEMA_COLUMNS_CACHE: Optional[set[tuple[str, str]]] = None


# ── Service ──────────────────────────────────────────────────────────────────

@dataclass
class NL2SQLTool:
    memory: Optional[NL2SQLMemory] = None
    embeddings: Optional[EmbeddingsService] = None
    client: Optional[openai.OpenAI] = None
    model: str = OPENAI_MODEL
    row_limit: int = NL2SQL_RESULT_ROW_LIMIT
    statement_timeout_ms: int = NL2SQL_STATEMENT_TIMEOUT_MS
    max_repair_rounds: int = NL2SQL_MAX_REPAIR_ROUNDS
    readonly_dsn: Optional[str] = None
    ensure_builtin_guidance: bool = True

    def __post_init__(self) -> None:
        self.memory = self.memory or NL2SQLMemory()
        self.embeddings = self.embeddings or EmbeddingsService()
        self.client = self.client or openai.OpenAI(api_key=OPENAI_API_KEY)
        self.readonly_dsn = (self.readonly_dsn or POSTGRES_READONLY_DSN
                             or POSTGRES_DSN)

    # ── Public ─────────────────────────────────────────────────────────────────

    def answer(
        self,
        nl_query: str,
        topic_id_hints: Optional[list[dict]] = None,
        claim_id_hints: Optional[list[dict]] = None,
    ) -> SQLOutput:
        t0 = time.monotonic()
        out = SQLOutput(nl_query=nl_query)

        if not nl_query or not nl_query.strip():
            out.attempts.append(SQLAttempt(sql="", error="empty_query",
                                            error_kind="sql_syntax"))
            out.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return out

        # 1. Recall context from Chroma 2
        self._ensure_builtin_guidance()
        embedding = self.embeddings.embed(nl_query)
        guidance_hits = self.memory.recall_guidance(embedding, n_results=8)
        schema_hits = self.memory.recall_schema(embedding, n_results=8)
        schema_hits = self._filter_live_schema_hints(schema_hits)
        success_hits = self.memory.recall_success(embedding, n_results=5)
        error_hits = self.memory.recall_errors(embedding, n_results=3)

        out.used_schema_hints = [h["id"] for h in schema_hits]
        out.used_examples = [h["id"] for h in success_hits]
        out.used_error_lessons = [h["id"] for h in error_hits]

        prompt_user = self._build_user_prompt(
            nl_query, guidance_hits, schema_hits, success_hits, error_hits,
            topic_id_hints=topic_id_hints,
            claim_id_hints=claim_id_hints,
        )

        # 2. Generate + execute + repair loop
        rounds = 0
        last_error: Optional[str] = None
        last_kind: Optional[str] = None

        while rounds <= self.max_repair_rounds:
            sql_raw = self._llm_generate(prompt_user, last_error, last_kind)
            safe_sql, syntax_kind = _sanitise_sql(sql_raw, self.row_limit)
            if syntax_kind is not None:
                attempt = SQLAttempt(sql=sql_raw or "",
                                      error="sanitiser rejected",
                                      error_kind=syntax_kind)
                out.attempts.append(attempt)
                last_error = "sanitiser rejected the SQL"
                last_kind = syntax_kind
                rounds += 1
                continue

            attempt = SQLAttempt(sql=safe_sql)
            try:
                rows, columns = self._execute(safe_sql)
                attempt.rows_returned = len(rows)
                out.attempts.append(attempt)
                out.final_sql = safe_sql
                out.rows = rows
                out.columns = columns
                out.success = True
                # Success-loop validation (PROJECT_REDESIGN_V2.md 7b-(3)):
                if attempt.rows_returned == 0:
                    # Soft signal: empty result. Do not retry; surface
                    # error_kind so Critic can decide.
                    attempt.error_kind = "sql_empty_result"
                if attempt.rows_returned >= self.row_limit:
                    # Hit cap; flag a soft warning but treat as success.
                    attempt.error_kind = "sql_limit_hit"
                break
            except _SQLExecutionError as exc:
                attempt.error = exc.detail[:240]
                attempt.error_kind = exc.kind
                out.attempts.append(attempt)
                last_error = exc.detail[:240]
                last_kind = exc.kind
                rounds += 1
                # Persist error lesson on third (final) failure
                if rounds > self.max_repair_rounds:
                    self._record_error_lesson(nl_query, safe_sql, exc, embedding)
                continue

        out.elapsed_ms = int((time.monotonic() - t0) * 1000)
        # Day 8 metrics
        try:
            from services.metrics import metrics
            metrics.inc("nl2sql.calls",
                        labels={"success": str(out.success).lower()})
            metrics.observe("nl2sql.repair_rounds", len(out.attempts) - 1)
            metrics.observe("nl2sql.latency_ms", out.elapsed_ms)
            metrics.observe("nl2sql.rows_returned", len(out.rows))
        except Exception:
            pass
        log.info("nl2sql.answer_done",
                 nl=nl_query[:60],
                 success=out.success,
                 attempts=len(out.attempts),
                 rows=len(out.rows),
                 elapsed_ms=out.elapsed_ms)
        return out

    def record_success(self, nl_query: str, sql: str) -> str:
        """External callers (Critic/Reflection) record a clean success."""
        embedding = self.embeddings.embed(nl_query)
        return self.memory.upsert_success(nl_query, sql, embedding)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_user_prompt(
        self, nl_query: str,
        guidance_hits: list[dict],
        schema_hits: list[dict],
        success_hits: list[dict],
        error_hits: list[dict],
        topic_id_hints: Optional[list[dict]] = None,
        claim_id_hints: Optional[list[dict]] = None,
    ) -> str:
        sections: list[str] = [f"Question: {nl_query}"]
        if topic_id_hints:
            # Tell the LLM exactly which topic_ids semantically match.
            # The system prompt says: prefer WHERE topic_id IN (...) over
            # label LIKE matching when these are provided.
            hint_lines = [
                f"  - {h['topic_id']!r} (label={h.get('label','')!r}, "
                f"similarity={h.get('similarity', 0):.2f})"
                for h in topic_id_hints
            ]
            sections.append(
                "Pre-resolved topic_ids (use these instead of label matching):"
            )
            sections.extend(hint_lines)
        if claim_id_hints:
            # Hybrid claim_search has already located semantically matching
            # claims_v2 rows. Use these claim_ids directly instead of
            # writing a fresh embedding/tsvector match in SQL.
            hint_lines = [
                f"  - {h['claim_id']!r} (text={h.get('claim_text','')[:90]!r}, "
                f"score={h.get('score', 0):.3f})"
                for h in claim_id_hints
            ]
            sections.append(
                "Pre-resolved claim_ids from hybrid claim_search "
                "(use WHERE c.claim_id IN (...) on claims_v2 instead of "
                "matching claim phrasing manually):"
            )
            sections.extend(hint_lines)
        if guidance_hits:
            sections.append("NL2SQL guidance from Chroma 2:")
            for h in guidance_hits:
                sections.append(f"  - {h.get('document', '')}")
        if schema_hits:
            sections.append("Schema hints:")
            for h in schema_hits:
                sections.append(f"  - {h.get('document', '')}")
        if success_hits:
            sections.append("Successful examples:")
            for h in success_hits:
                sections.append(f"  - {h.get('document', '')}")
        if error_hits:
            sections.append("Avoid these mistakes:")
            for h in error_hits:
                sections.append(f"  - {h.get('document', '')}")
        return "\n".join(sections)

    def _filter_live_schema_hints(self, hits: list[dict]) -> list[dict]:
        """Drop stale Chroma 2 schema docs for columns no longer in Postgres."""
        live = self._live_schema_columns()
        if not live:
            return hits
        filtered: list[dict] = []
        for hit in hits:
            meta = hit.get("metadata") or {}
            table = str(meta.get("table_name") or "").strip()
            column = str(meta.get("column_name") or "").strip()
            if not table or not column:
                # Non-column schema prose is still useful.
                filtered.append(hit)
                continue
            if (table, column) in live:
                filtered.append(hit)
            else:
                log.warning(
                    "nl2sql.stale_schema_hint_skipped",
                    table=table,
                    column=column,
                    hint_id=hit.get("id"),
                )
        return filtered

    def _live_schema_columns(self) -> set[tuple[str, str]]:
        global _LIVE_SCHEMA_COLUMNS_CACHE
        if _LIVE_SCHEMA_COLUMNS_CACHE is not None:
            return _LIVE_SCHEMA_COLUMNS_CACHE
        try:
            conn = psycopg2.connect(self.readonly_dsn)
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT table_name, column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                        """
                    )
                    _LIVE_SCHEMA_COLUMNS_CACHE = {
                        (str(table), str(column))
                        for table, column in cur.fetchall()
                    }
                    self.memory.prune_stale_schema(_LIVE_SCHEMA_COLUMNS_CACHE)
        except Exception as exc:
            log.warning("nl2sql.live_schema_load_failed",
                        error=str(exc)[:160])
            _LIVE_SCHEMA_COLUMNS_CACHE = set()
        finally:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass
        return _LIVE_SCHEMA_COLUMNS_CACHE

    def _ensure_builtin_guidance(self) -> None:
        """Seed durable NL2SQL operating rules into Chroma 2 if missing."""
        if not self.ensure_builtin_guidance:
            return
        global _BUILTIN_GUIDANCE_SEEDED
        if _BUILTIN_GUIDANCE_SEEDED:
            return

        for rule_id, text, category, priority in _BUILTIN_GUIDANCE:
            try:
                self.memory.upsert_guidance(
                    rule_id=rule_id,
                    text=text,
                    embedding=self.embeddings.embed(text),
                    category=category,
                    priority=priority,
                )
            except Exception as exc:
                log.warning("nl2sql.guidance_seed_failed",
                            rule_id=rule_id,
                            error=str(exc)[:120])
        _BUILTIN_GUIDANCE_SEEDED = True

    def _llm_generate(
        self,
        user_prompt: str,
        prev_error: Optional[str],
        prev_kind: Optional[str],
    ) -> str:
        if prev_error:
            user_prompt = (
                f"{user_prompt}\n\nPrevious attempt failed (kind={prev_kind}): "
                f"{prev_error}\nWrite a corrected query."
            )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _GENERATE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "{}").strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return ""
            return (data.get("sql") or "").strip()
        except Exception as exc:
            log.error("nl2sql.llm_error", error=str(exc)[:120])
            return ""

    def _execute(self, sql: str) -> tuple[list[dict], list[str]]:
        try:
            conn = psycopg2.connect(
                self.readonly_dsn,
                cursor_factory=psycopg2.extras.RealDictCursor,
            )
        except Exception as exc:
            raise _SQLExecutionError("sql_connection", str(exc))
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET LOCAL statement_timeout = {self.statement_timeout_ms}")
                    cur.execute("SET TRANSACTION READ ONLY")
                    try:
                        cur.execute(sql)
                    except psycopg2.errors.UndefinedColumn as exc:
                        raise _SQLExecutionError("sql_unknown_column", str(exc))
                    except psycopg2.errors.UndefinedTable as exc:
                        raise _SQLExecutionError("sql_unknown_column", str(exc))
                    except psycopg2.errors.SyntaxError as exc:
                        raise _SQLExecutionError("sql_syntax", str(exc))
                    except psycopg2.errors.QueryCanceled as exc:
                        raise _SQLExecutionError("sql_timeout", str(exc))
                    except psycopg2.errors.DataError as exc:
                        raise _SQLExecutionError("sql_type_mismatch", str(exc))
                    except psycopg2.Error as exc:
                        raise _SQLExecutionError("sql_other", str(exc))
                    rows = list(cur.fetchall())
                    columns = ([d[0] for d in cur.description]
                               if cur.description else [])
                    return rows, columns
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _record_error_lesson(
        self,
        nl_query: str,
        sql: str,
        exc: "_SQLExecutionError",
        embedding: list[float],
    ) -> None:
        """Push a curated error pattern into Chroma 2 (kind=error)."""
        if not sql or not sql.strip():
            # Empty SQL means the LLM failed to generate anything — there's
            # no bad SQL pattern to learn from.  Recording "(no SQL produced)"
            # creates a misleading lesson that tells future LLM calls to avoid
            # writing SQL for this type of question, forming a failure loop.
            log.warning("nl2sql.skip_empty_sql_lesson", nl=nl_query[:80])
            return
        try:
            bad_pattern = sql
            self.memory.upsert_error(
                failure_reason=f"{exc.kind}: {exc.detail[:200]}",
                bad_pattern=f"NL: {nl_query}\nSQL: {bad_pattern}",
                embedding=embedding,
            )
        except Exception as record_exc:
            log.error("nl2sql.error_record_failed",
                      error=str(record_exc)[:120])


@dataclass
class _SQLExecutionError(Exception):
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail[:200]}"
