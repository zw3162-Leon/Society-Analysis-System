# Society Analysis System

A retrieval-augmented analysis system over **Reddit community discussions** and
**authoritative news sources** (AP / Reuters / BBC / NYT). Users ask questions
in natural language; the system fans the request out to up to three retrieval
branches in parallel, composes a citation-bearing markdown report with a
quality critic guarding hallucination, and feeds critic verdicts back into a
self-curating reflection store.

The pre-loaded data snapshot lets a grader use the system end-to-end without
running ingestion. See [§3 How to Run](#3-how-to-run) for the exact contents.

---

## 1. System Structure

### 1.1 End-to-end chat flow

```
POST /chat/query
   │
   ▼
ChatOrchestrator (agents/chat_orchestrator.py)
   │
   ├─ 1. QueryRewriter         (agents/query_rewriter.py)
   │       Splits the user message into 1–3 atomic Subtasks; classifies each
   │       intent (fact_check / topic_claim_audit / community_count /
   │       community_listing / trend / propagation_trace / influencer_query /
   │       coordination_check / community_structure / cascade_query / ...).
   │       Pulls negative few-shot from Chroma 3 (past route violations).
   │
   ├─ 2. PlanVerifier           (agents/plan_verifier.py)
   │       Deterministic rule check on the Rewriter output. Records every
   │       correction back to Chroma 3 as an anti-pattern.
   │
   ├─ 3. BoundedPlannerV2       (agents/planner_v2.py)
   │       For each subtask:
   │         • TopicResolver       — semantic match user phrase → topic_id
   │         • ClaimSearchTool     — hybrid pgvector + simhash + tsvector
   │                                 over claims_v2 (only for fact_check /
   │                                 topic_claim_audit subtasks)
   │         • _scrub_official_source_mentions — strip "using AP/BBC/..."
   │                                 phrases from the NL2SQL nl_query
   │         • Dispatch to up to 3 branch runners in parallel; ≤5 total
   │           branch calls per workflow
   │
   ├─ 3a. Evidence (RAG) branch   (tools/hybrid_retrieval.py)
   │       Retrieves authoritative news chunks from Chroma 1
   │       (chroma_official). Pipeline per call:
   │         1. metadata pre-filter (source / domain / tier / title /
   │            topic_hint); LLM-friendly key aliasing fixes mistakes like
   │            `official_sources` → `source`.
   │         2. Dense recall via OpenAI text-embedding-3-small + cosine.
   │         3. BM25 recall (rank_bm25, in-memory over the filtered subset).
   │         4. Reciprocal Rank Fusion (k=60).
   │         5. Optional bge-reranker-base rerank for the top candidates.
   │         6. Return EvidenceBundle with chunks + Citation
   │            (title / outlet / URL / publish_date when available).
   │
   ├─ 3b. NL2SQL branch           (tools/nl2sql_tools.py)
   │       Generates and executes a safe read-only Postgres SELECT against
   │       posts_v2 / topics_v2 / entities_v2 / claims_v2 / post_claims_v2.
   │         1. Embed the nl_query and recall four kinds of records from
   │            Chroma 2 (`recall_guidance`, `recall_schema`,
   │            `recall_success`, `recall_errors`).
   │         2. Build the system prompt with: pre-resolved topic_id_hints
   │            from TopicResolver, pre-resolved claim_id_hints from
   │            ClaimSearchTool, schema hints, success exemplars,
   │            error_lessons, durable guides.
   │         3. LLM emits one SELECT (no semicolons, no DML).
   │         4. Sanitiser: forces a LIMIT, strips DML, blocks multi-statement.
   │         5. Execute via POSTGRES_READONLY_DSN with statement_timeout.
   │         6. Self-repair loop up to 3 rounds; if every round fails,
   │            persist the failure as a Chroma 2 error_lesson.
   │       Special path: `topic_claim_audit` intent runs a deterministic
   │       SELECT against claims_v2 + post_claims_v2 (skipping the LLM)
   │       so the Report Writer always gets canonical claim text instead
   │       of having to mine it out of comment bodies.
   │
   ├─ 3c. Knowledge Graph branch  (tools/kg_query_tools.py +
   │                                tools/kg_analytics.py)
   │       Routes the subtask intent to a Kuzu / NetworkX analytic and
   │       returns nodes + edges + metrics (KGOutput):
   │         • propagation_trace   → Cypher reply-chain traversal between
   │                                 two account ids (orphan-chain filter
   │                                 keeps the path narratable).
   │         • influencer_query    → NetworkX PageRank on the topic
   │                                 subgraph (real influence, not raw
   │                                 post counts).
   │         • coordination_check  → Louvain community detection.
   │         • community_structure → modularity / echo-chamber score.
   │         • cascade_query       → recursive reply-tree depth ranking.
   │         • topic_correlation   → shared-entity bridge between topics.
   │       Subgraph extraction is LRU-cached (services/kg_cache.py); writes
   │       bump a sequence so reads see fresh edges.
   │
   ├─ 4. ReportWriter           (agents/report_writer.py)
   │       LLM composes a markdown answer with inline citations from the
   │       branch outputs.
   │
   ├─ 5. QualityCritic          (agents/quality_critic.py)
   │       4-axis check: citation_completeness, numeric_consistency,
   │       on_topic, hallucination. One retry on failure; flips
   │       needs_human_review on second failure.
   │
   └─ 6. ReflectionStore        (services/reflection_store.py)
           Owns the closed loop. Three concurrent jobs every turn:

           A. Audit trail
              Writes one row per turn to PostgreSQL `reflection_log`
              (occurred_at, session_id, user_message, error_kind,
              failed_branch, causal_record_ids, payload). Source of
              truth for evaluating system behaviour over time.

           B. Ablation (only when critic verdict failed)
              For each causal_record_id reported by the critic, re-runs
              the failing step without that record. If the failure
              disappears, the record is "guilty" and gets deleted from
              its owning store:
                • record_ids starting with `schema::` / `success::` /
                  `error::` → Chroma 2 (NL2SQL memory).
                • record_ids starting with `module_card::` /
                  `workflow_*` / `composition_error::` → Chroma 3
                  (Planner memory).
              Anti-thrash: a record_id deleted+rewritten more than 2×
              in 24h is quarantined for 24h.

           C. Negative lesson write-back (only when critic verdict failed)
              Routes the error_kind to the right store so the next time
              the same question comes in, the Rewriter / Planner / NL2SQL
              sees the past failure as negative few-shot:

              error_kind                  → destination
              ────────────────────────────────────────────────────────
              sql_empty_result            → Chroma 2 `error` record
                                            (NL2SQL avoids the same
                                            zero-result pattern)
              missing_branch              → Chroma 3 `workflow_error`
              wrong_branch_combo          → Chroma 3 `workflow_error`
                                            (Rewriter pulls these as
                                            route-violation few-shot)
              off_topic                   → Chroma 3 `workflow_error` or
                                            `composition_error` depending
                                            on which branch was at fault
              citation_missing            → Chroma 3 `composition_error`
              numeric_mismatch            → Chroma 3 `composition_error`
                                            (Writer learns to back every
                                            number with a branch row)

           Stores curated by ReflectionStore:

           Chroma 2 (chroma_nl2sql) — feeds the NL2SQL branch
              kind=schema   per-column descriptions (mirrored from
                            Postgres schema_meta by SchemaSync).
              kind=guide    durable rules (e.g. "posts_v2.source is
                            always 'reddit'"; "use claim_search for
                            fact-check, not posts_v2.text_tsv").
              kind=success  successful (NL, SQL) exemplars, embedded
                            and re-ranked by Critic verdict count.
              kind=error    SQL patterns that produced empty results
                            or runtime errors — used as negative
                            few-shot in the next NL2SQL prompt.

           Chroma 3 (chroma_planner) — feeds the Planner & Rewriter
              kind=module_card        per-branch capability cards
                                      (when_to_use / when_not_to_use /
                                      input_schema / output_schema).
              kind=workflow_success   (question, branches_used) pairs
                                      that produced clean Critic
                                      verdicts; bumps a `confidence`
                                      counter when reused.
              kind=workflow_error     route-violation anti-patterns
                                      (e.g. "fan-out to evidence on a
                                      pure community_listing question");
                                      surfaced as negative few-shot to
                                      the Rewriter on the next call.
              kind=composition_error  Writer-attribution failures
                                      (citation_missing, numeric_mismatch)
                                      so future Writer prompts know the
                                      exact prose pattern that broke.
```

### 1.2 Three retrieval branches

| Branch | Tool | Backed by | Strengths |
|---|---|---|---|
| **Evidence** | `tools/hybrid_retrieval.py` | Chroma 1 (`chroma_official`) | Authoritative reporting; dense + BM25 + RRF + bge-reranker-base |
| **NL2SQL** | `tools/nl2sql_tools.py` + `tools/claim_search.py` | PostgreSQL 16 + pgvector | Counts / filters / aggregations / claim listings; semantic claim search |
| **Knowledge Graph** | `tools/kg_query_tools.py` | Kuzu graph DB | Multi-hop reply traversal, PageRank, Louvain communities, modularity / echo-chamber detection |

### 1.3 Data stores

| Store | What lives there | Persistence |
|---|---|---|
| **PostgreSQL 16 (pgvector)** | `posts_v2`, `topics_v2`, `entities_v2`, `post_entities_v2`, **`claims_v2`** (atomic claims with embedding + simhash + tsvector), **`post_claims_v2`** (M:N), `schema_meta`, `reflection_log` | Docker named volume `postgres_data` |
| **Chroma 1** (`chroma_official`) | Authoritative news chunks (AP / Reuters / BBC / NYT) | `./data/chroma` (bind-mount) |
| **Chroma 2** (`chroma_nl2sql`) | NL2SQL schema docs + success exemplars + error_lessons + durable guides | `./data/chroma` |
| **Chroma 3** (`chroma_planner`) | Planner module cards + workflow exemplars + route-violation anti-patterns | `./data/chroma` |
| **Kuzu** | `Account / Post / Topic / Entity` nodes; `Posted / Replied / BelongsToTopic / HasEntity` edges | `./data/kuzu_graph` |

### 1.4 Offline ingestion pipeline (precompute)

```
RedditService
   ├─ fetch_posts            (worldnews scrape)
   ├─ ingest                 (multimodal + entities + simhash dedup)
   ├─ normalize
   ├─ emotion_baseline       (per-post fear / anger / hope / disgust / neutral)
   ├─ topic_cluster          (KMeans on OpenAI embeddings; chunked ≤2000)
   ├─ extract_claims         (LLM extracts atomic claims from submission
   │                          titles; simhash-deduped; comments link to
   │                          parent submission's claims)
   ├─ schema_propose         (Schema-aware Agent → schema_meta + Chroma 2)
   └─ persist_v2             (PostgreSQL + Kuzu writes)
```

The **claim extraction stage** is the system's answer to the "fact-check
finds zero rows" failure mode: claim text used to live only in Reddit
submission titles (not in the `posts_v2.text` of the comments that
discussed them). `claims_v2` now stores each atomic claim once, with a
1536-dim pgvector embedding, a 64-bit Charikar simhash, and a tsvector,
enabling hybrid semantic + lexical retrieval.

### 1.5 Repository layout

```
agents/         Pipeline + chat agents (rewriter, planner, writer, critic,
                claim_extractor, ...)
tools/          Atomic operations (hybrid_retrieval, nl2sql, claim_search,
                topic_resolver, kg_query_tools)
services/       Storage + LLM wrappers (postgres, chroma, kuzu, embeddings,
                nl2sql_memory, planner_memory, reflection_store)
models/         Pydantic data contracts
api/            FastAPI routes (chat, retrieve, runs, reflection, plan,
                admin/import, admin/nl2sql, artifacts, health)
ui/             Single-page Streamlit chat UI
db/             schema_v2.sql (auto-applied on Postgres first boot)
config/         YAML configs (official_sources.yaml)
scripts/        Seed + maintenance scripts (seed_planner_memory,
                seed_emotion_nl2sql_examples, seed_claims_nl2sql_examples,
                bootstrap.sh, ...)
eval/           9-module evaluation suite
data/           Persisted Chroma + Kuzu + run artifacts (bind-mounted into
                the api / ui containers)
```

---

## 2. Setup Instructions

### 2.1 Prerequisites

- **Docker Desktop** with Compose v2 (tested on Docker 29.x).
- **An OpenAI API key** with access to `gpt-4o` and `text-embedding-3-small`.
  All LLM calls (rewriter / NL2SQL / writer / critic / claim extractor) and
  all embeddings go through OpenAI.
- **~10 GB free disk** for the image, model cache, and data volume.

That's it — you do **not** need a local PostgreSQL, Python virtualenv, or
any JVM. The entire stack runs in three Docker containers.

### 2.2 Step-by-step setup

1. **Clone the repo**:

   ```bash
   git clone https://github.com/yuchangyuan1/society-analysis-system
   cd society-analysis-system
   ```

   The repo ships the pre-loaded demo snapshot:
   - `db/snapshot_data.sql.gz` — Postgres data (auto-restored on first boot).
   - `data/chroma/` — three Chroma collections (HNSW indexes).
   - `data/kuzu_graph` — Kuzu graph DB.
   - `data/official_chunks/` — JSONL of fetched RSS items.

   **Do not delete any of these** unless you intend to re-ingest from
   scratch.

2. **Create `.env`** from the template and fill in your OpenAI key:

   ```bash
   cp .env.example .env
   ```

   Open `.env` and replace `OPENAI_API_KEY=sk-...` with your real key.
   Every other variable has a working default.

3. **Bring the stack up**:

   ```bash
   docker compose up -d --build
   ```

   The first build pulls the pgvector PostgreSQL image and the Python
   deps (torch, sentence-transformers, openai, chromadb, kuzu, ...).
   Total cold-start ≈ 5–10 min; warm restarts are seconds.

   **What happens automatically on first boot:**
   - PostgreSQL initdb runs `db/schema_v2.sql` (creates tables / triggers
     / indexes / pgvector extension), then loads
     `db/snapshot_data.sql.gz` (~2 000 posts, 20 topics, 62 claims).
   - The api / ui containers come up healthy and serve the snapshot.
   - `bge-reranker-base` (~600 MB) downloads on the first chat call that
     uses Evidence retrieval; subsequent calls use the cache.

4. **Verify the stack is healthy**:

   ```bash
   docker compose ps
   ```

   All three services (`postgres`, `api`, `ui`) should be `Up` and `healthy`.

   ```bash
   curl http://127.0.0.1:8000/health
   ```

   Expected: `{"ok": true, "runs_root": "/app/data/runs"}`.

   Sanity-check the snapshot loaded:

   ```bash
   docker compose exec postgres psql -U society -d society_db \
     -c "SELECT COUNT(*) AS posts FROM posts_v2;" \
     -c "SELECT COUNT(*) AS topics FROM topics_v2;" \
     -c "SELECT COUNT(*) AS claims FROM claims_v2;"
   ```

   Expected counts: posts ≈ 2 072, topics = 20, claims = 62.

### 2.3 Optional resets

```bash
# Stop containers but keep all data (safe).
docker compose down

# Full nuclear reset — DROPS the Postgres volume + the model cache.
# Use only if you intend to re-ingest from scratch.
docker compose down -v
```

---

## 3. How to Run

### 3.1 Web UI (primary interface)

Open <http://localhost:8501> in a browser. The page is a single-page
chat against the pre-loaded snapshot.

**The sidebar** lets you:
- Pick subreddits and date ranges for fresh imports;
- Pick official sources (AP / Reuters / BBC / NYT) and import their RSS feeds;
- Toggle `Append` / `Overwrite` mode;
- Toggle "Show raw technical output" to see per-branch JSON in each answer.

The main chat panel shows the markdown report, citation list, route module
cards (RAG / Knowledge Graph / NL2SQL), and an interactive KG visualisation
when the Knowledge Graph branch returns nodes/edges.

### 3.2 REST API (for direct probing or scripting)

Base URL: <http://127.0.0.1:8000>. Interactive OpenAPI docs at
<http://127.0.0.1:8000/docs>.

Useful endpoints:

```bash
# Full chat (the orchestrator: rewriter → verifier → planner → writer → critic).
curl -X POST http://127.0.0.1:8000/chat/query \
     -H 'Content-Type: application/json' \
     -d '{"session_id":"demo","message":"What topics are trending and who is amplifying them?"}'

# Direct branch access (bypasses the orchestrator):
curl -X POST http://127.0.0.1:8000/retrieve/nl2sql \
     -H 'Content-Type: application/json' \
     -d '{"nl_query":"How many posts per dominant emotion?"}'

curl -X POST http://127.0.0.1:8000/retrieve/evidence \
     -H 'Content-Type: application/json' \
     -d '{"query":"Trump tariffs on EU"}'

curl -X POST http://127.0.0.1:8000/retrieve/kg \
     -H 'Content-Type: application/json' \
     -d '{"query_kind":"key_nodes","target":{"top_k":5}}'

# Inspect a session.
curl http://127.0.0.1:8000/chat/session/demo

# Run artifacts (everything written under data/runs/{run_id}/).
curl http://127.0.0.1:8000/runs
curl http://127.0.0.1:8000/runs/<run_id>/report

# Reflection store inspector.
curl 'http://127.0.0.1:8000/reflection/log?limit=20'
curl 'http://127.0.0.1:8000/reflection/chroma2?kind=success&limit=20'
```

### 3.3 Pre-loaded data snapshot

The shipped `data/` folder contains a working dataset so the grader can
use the system without waiting on ingestion or burning OpenAI tokens.

| Store | Snapshot |
|---|---|
| `posts_v2` | **2 072 Reddit posts** from r/worldnews (50 link-post submissions + 2 022 comments) |
| `topics_v2` | **20 topics** from KMeans clustering, e.g. *Concerns Over FIFA World Cup 2026 Hosting*, *US-Iran Tensions Amid Ceasefire Talks*, *Concerns Over New Virus Outbreak*, *Criticism of Presidential Leadership and Policies* |
| `claims_v2` | **62 atomic claims** extracted from submission titles, deduped by simhash |
| `post_claims_v2` | **2 594 post → claim links** (claim → discussing comments) |
| `chroma_official` | **142 chunks** — AP=20 / Reuters=20 / BBC=62 / NYT=40 |
| Kuzu | Account / Post / Topic / Entity nodes + Posted / Replied / BelongsToTopic / HasEntity edges, derived from the same posts |

**Honest disclosure on date ranges.** The Reddit fetch and the official-RSS
fetch both pull *current* items, not historical archives:

- Reddit's public scraping returns currently-hot content. The snapshot's
  `posts_v2.posted_at` values are **2026-05-07 to 2026-05-08**, regardless
  of the date range that was originally requested when the snapshot was
  produced. (Reddit doesn't time-travel; this is an intrinsic limit of the
  scraping path the project uses, documented in code.)
- The official-source crawler reads each outlet's current RSS feed. The
  142 chunks are whatever was on AP / Reuters / BBC / NYT's RSS at the time
  of fetching. The crawler exposes a date-range field to the UI but
  emits a warning that historical backfill is not implemented:
  *"Official RSS import uses the configured feeds' current items..."*

The snapshot is therefore a **frozen point-in-time view of the live web on
2026-05-08**. Every example question in §4 is verified to return useful
results against this snapshot.

### 3.4 Re-importing fresh data (optional — DO NOT use for grading)

> ⚠️ **For grading: please test against the pre-loaded snapshot.**
> The Example questions in §4 are written against the specific topics,
> claims, and accounts in the shipped data. If you click `Import` in
> the UI sidebar, the system runs a fresh Reddit + RSS fetch and
> **overwrites the demo snapshot with whatever is currently trending on
> Reddit**, at which point the example prompts will refer to topics /
> claims / accounts that no longer exist. Reddit's public scraping
> cannot time-travel back to the snapshot date, so the original data
> cannot be regenerated by clicking Import again.
>
> If the UI's Import controls were accidentally used, restore the
> snapshot with:
>
> ```bash
> docker compose down -v   # drop the modified Postgres volume
> git checkout -- data/    # revert Chroma + Kuzu files
> docker compose up -d     # initdb auto-restores db/snapshot_data.sql.gz
> ```

If you want a different snapshot, the UI sidebar lets you pick
subreddits + sources, set `Overwrite` mode, check the confirmation
box, and click `Import`. The job runs in the background; check
`docker compose logs -f api` or `GET /admin/import/jobs/{job_id}` for
progress. A Reddit overwrite of ~2 000 posts takes ~10 min and
~$0.50 of OpenAI credit (claim extraction + embeddings).

### 3.5 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `docker compose up` fails with port-in-use error | Host port 8000 / 8501 / 15432 already taken | Stop the conflicting service, or remap in `docker-compose.yml` (e.g. change `"8000:8000"` to `"8010:8000"` and update `RESEARCH_API_BASE` in `.env`). |
| `/health` returns `{"ok": false}` | Snapshot didn't auto-restore (initdb only runs on an empty Postgres volume) | Stop the stack, `docker compose down -v` to drop the volume, then `docker compose up -d --build` to re-init from `db/snapshot_data.sql.gz`. |
| `posts_v2` is empty after first boot | Same as above | Same fix. |
| `/chat/query` answers say *"I couldn't gather enough data"* | Bad / missing OpenAI key, or empty stores | Verify `.env`'s `OPENAI_API_KEY`; sanity-check counts (§2.2 step 4). |
| `branches_used: []` for every query | OpenAI rejecting requests | Check API key validity & rate limits; restart api with `docker compose restart api`. |
| First chat call is slow (~30 s) | bge-reranker-base model downloading on demand | One-time; subsequent calls use the cached model in the `model_cache` volume. |
| `needs_human_review: true` on a factual answer | Quality Critic flagged numeric_consistency or hallucination | The system flagging itself is by design; inspect `branch_outputs` in the response to see what it caught. |
| Reuters / AP RSS returning 0 chunks | Outlet's RSS unreachable from your network | Already proxied via Google News in `config/official_sources.yaml`; if all four return zero, the snapshot still has the shipped 142 chunks. |

---

## 4. Example Usage

Seven sample questions, each verified end-to-end against the shipped
snapshot. Paste the prompt into the chat at <http://localhost:8501>
or `POST /chat/query` to see the real response.

> ⚠️ **Test against the pre-loaded snapshot — do NOT click `Import`
> in the UI sidebar before running these.** The questions reference
> specific topics / claims / account names that exist only in the
> shipped data. Importing will overwrite the snapshot with current
> Reddit trends, breaking every example below. See §3.4 for restoration
> if this happens.

### Example 1 — Trending topic listing (NL2SQL only)

> *List the main topics in the selected Reddit data, and summarize discussion volume, dominant emotion, and notable shifts.*

### Example 2 — Topic emotion + representative posts (NL2SQL + KG)

> *For the topic about Concerns Over New Virus Outbreak, summarize the dominant emotions, representative posts, and how sentiment differs across discussion clusters.*

### Example 3 — Topic-level propagation (KG only)

> *For the topic about Casualties in Russia-Ukraine Conflict, trace propagation paths or reply chains and explain what the Knowledge Graph shows.*

### Example 4 — Account-to-account propagation path (KG only)

> *Trace the propagation path between GeneReddit123 and GiveMeSomeSunshine3.*

### Example 5 — Topic amplifier ranking (KG + NL2SQL)

> *For the topic about Casualties in Russia-Ukraine Conflict, identify key amplifying accounts and explain the graph evidence behind the ranking.*

### Example 6 — Topic claim audit (NL2SQL + Evidence)

> *For the topic about US-Iran Tensions Amid Ceasefire Talks, list Reddit claims and classify which are consistent with official/evidence sources, which contradict them, and which lack enough evidence.Include author, verdict, the official/evidence statement, and citation.*

### Example 7 — Single-claim fact-check (Evidence + NL2SQL)

> *Fact-check this Reddit claim using the selected official/evidence sources: Trump threatens 'much higher' tariffs on EU by July 4.*


