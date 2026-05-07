# Society Analysis

A real-time retrieval system over **Reddit discussions** and
**official / evidence sources**. Users ask questions in natural language; the system fans
the request out to **three retrieval branches in parallel** —

- **Evidence Retrieval** — hybrid (Dense + BM25 + RRF + Rerank) over
  authoritative news (BBC / NYT / Reuters / AP / Xinhua)
- **NL2SQL** — natural language → safe Postgres SELECT against community
  posts, with semantic topic resolution and self-correcting repair loop
- **Knowledge Graph Query** — Cypher over a Kuzu graph of accounts /
  posts / topics / entities

— then composes a citation-bearing markdown report with a Quality Critic
guarding against hallucination, and feeds Critic verdicts back into a
Reflection store that auto-curates the system's own learned exemplars.

Long-running chat sessions stay performant: the conversation list is
window-bounded and older turns are auto-compressed into a rolling
summary, so a session can run for hundreds of turns without ballooning
memory or prompt size.

> Want the design rationale? See `PROJECT_REDESIGN_V2.md`.
> Want the code map? See `workflow.md`.

---

## What this is good for

```
"What topics are trending and who's amplifying them?"
"What did BBC and Reuters say about the Iran ceasefire?"
"Is the Reddit claim 'vaccines reduce hospitalisation by 90%' supported by official sources?"
"How many angry posts about climate this week, and what entities do they mention?"
"Compare the official line on the Cuba sanctions with what users are saying."
```

The system runs three branches concurrently when the question benefits from
cross-source synthesis, then a Report Writer assembles a markdown answer
with citations built from source metadata, including title, outlet/domain,
URL, and publication date when available. A Quality Critic checks every numerical claim against the SQL/KG row
it cites; if the LLM hallucinates a number, the response is flagged
`needs_human_review=True`.

---

## Knowledge Graph branch (Phase A→D + production hardening)

The KG branch isn't a SQL view in disguise — it owns five capabilities
that PG can't express:

| Query | Backed by | Use case |
|---|---|---|
| `propagation_path` | Kuzu Cypher (Replied chain, both directions) | "trace how this rumour spread between alice and dave" |
| `cascade_tree` | Kuzu Cypher (recursive descendants) | "show me the entire reply tree under this post" |
| `viral_cascade` | Kuzu + ranking | "longest / most-amplified cascades in topic T" |
| `influencer_rank` | NetworkX PageRank | "who's actually influential" (not just post count) |
| `bridge_accounts` | NetworkX betweenness | "who connects two opposing communities" |
| `coordinated_groups` | NetworkX Louvain | "is this organised posting" |
| `echo_chamber` | Modularity threshold | "is topic T a closed bubble" |

The Planner auto-routes 5 SubtaskIntents (`propagation_trace`,
`influencer_query`, `coordination_check`, `community_structure`,
`cascade_query`) to the right method. Subgraph extraction is LRU-cached;
freshness defaults to 30 days for trending / propagation queries.

## Quick start

### Option A. Docker Compose, recommended for demos

Use this path for course presentations or when another machine needs to
reproduce the system with minimal local setup.

**Prerequisites**

- Docker Desktop (Compose v2). Tested on Linux, macOS, and Windows
  (run the script from Git Bash or WSL).
- ~10 GB free disk for the image + model cache.
- An OpenAI API key.

**One-command bootstrap (recommended for first clone)**

```bash
cp .env.example .env          # edit OPENAI_API_KEY
LOAD_FIXTURE=1 ./scripts/bootstrap.sh
```

`scripts/bootstrap.sh` is idempotent and takes care of:

1. validating that `OPENAI_API_KEY` is filled in,
2. `docker compose up -d --build` (first build ~10 min, pulls torch + sentence-transformers),
3. waiting for `api` to report `healthy`,
4. seeding Chroma 3 (planner memory) via `scripts.seed_planner_memory`,
5. seeding Chroma 2 (NL2SQL emotion exemplars + guardrail guides) via
   `scripts.seed_emotion_nl2sql_examples`,
6. when `LOAD_FIXTURE=1`, loading `tests/fixtures/posts_v2_smoke.jsonl`
   so the UI has demo data on first launch.

When the script finishes, open <http://localhost:8501>.

**Manual equivalent**

If you prefer to run the steps yourself:

```bash
cp .env.example .env          # edit OPENAI_API_KEY
docker compose up -d --build
# wait until `docker compose ps` shows api as (healthy)
docker compose exec api python -m scripts.seed_planner_memory
docker compose exec api python -m scripts.seed_emotion_nl2sql_examples
# optional: load demo data
docker compose exec api python main.py --jsonl tests/fixtures/posts_v2_smoke.jsonl
```

**What the stack runs**

- `postgres` on host port `15432`; `db/schema_v2.sql` is applied on first boot;
- `api` (FastAPI) on <http://localhost:8000>, OpenAPI docs at `/docs`;
- `ui` (Streamlit) on <http://localhost:8501>;
- shared runtime data under `./data` (bind-mounted into both api and ui);
- persistent named volumes for Postgres data and the Hugging Face model cache.

**Useful operations**

```bash
# Pull official/evidence sources into Chroma 1.
docker compose exec api python -m agents.official_ingestion_pipeline --once

# Tail logs.
docker compose logs -f api ui

# Stop containers but keep all data.
docker compose down

# Full reset (drops Postgres volume and model cache too).
docker compose down -v
```

Inside containers, the database hostname is `postgres`, not `localhost`.
The Compose file injects the correct `POSTGRES_DSN`; keep `.env.example`
using `localhost` for non-Docker local runs.

### Option B. Local Python environment

Use this path if you want to run directly from `.venv`.

#### 1. Install

```bash
git clone https://github.com/zw3162-Leon/Society-Analysis-System
cd Society-Analysis-System
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e .[dev]
```

You also need:
- **Postgres ≥ 14** with the `pg_trgm` extension available (the schema
  installs it automatically).
- **OpenAI API key** (`OPENAI_API_KEY`) for LLM + embeddings.
- *(Optional)* **Anthropic API key** (`ANTHROPIC_API_KEY`) for the
  multimodal image-understanding agent. The pipeline degrades gracefully
  when it's missing.
- ~600 MB free for the bge-reranker-base model (downloaded on first use).

#### 2. Configure

Copy `.env.example` to `.env` (or just create `.env`):

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
POSTGRES_DSN=postgresql://society:society_pass@localhost:5432/society_db

# Optional
ANTHROPIC_API_KEY=sk-ant-...
POSTGRES_READONLY_DSN=postgresql://society_ro:...@localhost:5432/society_db
MULTIMODAL_DAILY_BUDGET_USD=5.0
```

#### 3. Initialize the database

```bash
# Create the database first if needed:
#   createdb society_db   (or: psql -c "CREATE DATABASE society_db;")

# Apply the v2 schema (6 tables + tsvector trigger + pg_trgm extension):
python -c "import psycopg2, config; \
  c = psycopg2.connect(config.POSTGRES_DSN); c.autocommit=True; \
  c.cursor().execute(open('db/schema_v2.sql').read())"

# If you're upgrading an existing PG that predates the lineage columns,
# run the idempotent migration once:
python -m scripts.migrate_run_lineage

# If you're upgrading an existing PG that predates the topic post-count
# trigger (any DB created before 2026-05), install the trigger and
# recompute every topics_v2.post_count once:
python -m scripts.migrate_topic_post_count_trigger
```

#### 4. Cold-start the planner memory (Chroma 3)

```bash
python -m scripts.seed_planner_memory
```

This loads three `ModuleCard`s (one per branch) and eight workflow
exemplars so the Planner has prior art to draw on.

#### 5. Run the offline ingestion (one-time, then schedule)

```bash
# Option A — quick smoke against bundled fixture (no network):
python main.py --jsonl tests/fixtures/posts_v2_smoke.jsonl

# Option B — pull live Reddit:
python main.py --subreddit conspiracy --days 3

# Pull authoritative-source articles (5 outlets):
python -m agents.official_ingestion_pipeline --once
```

After step 5 you should see:
- Postgres `posts_v2 / topics_v2 / entities_v2 / post_entities_v2` populated
- Kuzu graph at `data/kuzu_graph` with `Account → Post → Topic` edges
- Chroma collections at `data/chroma`:
  - `chroma_official` — authoritative news chunks (citation source)
  - `chroma_nl2sql` — schema descriptions + NL→SQL exemplars
  - `chroma_planner` — module cards + workflow exemplars

#### 6. Start the services

```bash
# Terminal 1 — FastAPI (port 8000)
uvicorn api.app:app --reload --port 8000

# Terminal 2 — Streamlit UI (port 8501)
streamlit run ui/streamlit_app.py
```

Open <http://localhost:8501>. The app is now a single-page Chat interface.
The sidebar lets you choose Reddit subreddits, official/evidence sources,
date ranges, and append/overwrite import mode.

---

## Try it from the UI

The sidebar contains editable suggested-question templates. They are written
without concrete topic names so you can fill in the topic or claim you want
to demonstrate.

Examples:

- `List the main topics in the selected Reddit data, and summarize discussion volume, dominant emotion, and notable shifts.`
- `For the topic about <your topic>, summarize the dominant emotions, representative posts, and how sentiment differs across discussion clusters.`
- `For the topic about <your topic>, trace propagation paths or reply chains and explain what the Knowledge Graph shows.`
- `For the topic about <your topic>, identify key amplifying accounts and explain the graph evidence behind the ranking.`
- `For the topic about <your topic>, list Reddit claims and classify which are consistent with official/evidence sources, which contradict them, and which lack enough evidence. Include author, verdict, the official/evidence statement, and citation.`
- `Fact-check this Reddit claim using the selected official/evidence sources: <paste claim here>.`

Each answer keeps the report readable by default, then shows:

- source citations when evidence was used;
- route-module cards for **RAG**, **Knowledge Graph**, and **NL2SQL**;
- a Knowledge Graph visualization when the KG branch returns nodes/edges;
- raw branch JSON only when **Show raw technical output** is enabled.

The sidebar import controls call the backend import API. `Append` adds new
pipeline output. `Overwrite` requires an explicit confirmation checkbox and
deletes retained data for the selected source scope before importing.

---

## Try it from the API

```bash
# Direct branch access (debug):
curl -X POST http://127.0.0.1:8000/retrieve/nl2sql \
  -H 'Content-Type: application/json' \
  -d '{"nl_query":"How many posts per dominant emotion?"}'

curl -X POST http://127.0.0.1:8000/retrieve/evidence \
  -H 'Content-Type: application/json' \
  -d '{"query":"US troop reduction in Germany"}'

curl -X POST http://127.0.0.1:8000/retrieve/kg \
  -H 'Content-Type: application/json' \
  -d '{"query_kind":"key_nodes","target":{"top_k":5}}'

# Full chat (all the orchestration):
curl -X POST http://127.0.0.1:8000/chat/query \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"What is trending and who is amplifying it?"}'

# Inspect a session:
curl http://127.0.0.1:8000/chat/session/demo

# Manual Reddit import (background job):
curl -X POST http://127.0.0.1:8000/admin/import/reddit \
  -H 'Content-Type: application/json' \
  -d '{"subreddits":["worldnews"],"start_date":"2026-05-02","end_date":"2026-05-03","mode":"append","limit_per_subreddit":100,"comment_limit":100,"include_comments":true}'

# Manual official/evidence import (background job):
curl -X POST http://127.0.0.1:8000/admin/import/official \
  -H 'Content-Type: application/json' \
  -d '{"sources":["ap","reuters","bbc","nyt"],"start_date":"2026-05-02","end_date":"2026-05-03","mode":"append","write_chroma":true}'

# Check import job status:
curl http://127.0.0.1:8000/admin/import/jobs/import_xxxxxxxxxxxx

# NL2SQL admin — inspect / clear bad error lessons from Chroma 2:
curl http://127.0.0.1:8000/admin/nl2sql/error-count
curl -X POST http://127.0.0.1:8000/admin/nl2sql/clear-errors

# Reflection inspector:
curl 'http://127.0.0.1:8000/reflection/chroma2?kind=success&limit=20'
curl 'http://127.0.0.1:8000/reflection/log?limit=20'

# Run artifacts (everything written under data/runs/{run_id}/):
curl http://127.0.0.1:8000/runs                       # list runs
curl http://127.0.0.1:8000/runs/<run_id>              # one run summary
curl http://127.0.0.1:8000/runs/<run_id>/report       # report.md
curl http://127.0.0.1:8000/runs/<run_id>/raw          # report_raw.json
curl http://127.0.0.1:8000/runs/<run_id>/metrics      # metrics.json
curl http://127.0.0.1:8000/runs/<run_id>/visual/<filename>   # counter_visuals/*
```

---

## Long-running sessions

Each chat session is stored as a single JSON file under `data/sessions/`.
The conversation list is automatically window-bounded so it stays small
and fast to load:

- Up to `SESSION_MAX_TURNS` (default **40**) turns kept verbatim.
- When that limit is exceeded, the oldest `SESSION_MIN_TURNS_TO_COMPACT`
  (default **10**) turns are compressed into a rolling `summary` field
  via one LLM call (`agents/conversation_compactor.py`).
- The Rewriter sees both the live window AND the rolling summary, so
  pronouns and topic anchors set hundreds of turns ago still resolve.
- LLM failures fall back to a plain trim, so persistence never breaks.

What you'll see in `data/sessions/<id>.json`:

```json
{
  "session_id": "demo",
  "current_topic_id": "topic_abc",
  "summary": "User asked about vaccine misinformation, ...",
  "summary_until_turn": 30,
  "archived_count": 30,
  "conversation": [/* most recent 40 turns */]
}
```

Override the window via env vars (see config table).

---

## Routine maintenance

```bash
# Daily decay sweep — drop stale experience records (kind=success / error /
# workflow_*). Anchor docs (kind=schema / module_card) are preserved.
python -m scripts.decay_chroma_experience

# Re-pull authoritative sources (BBC / NYT / Reuters / AP / Xinhua):
python -m agents.official_ingestion_pipeline --once

# Replay previously-written jsonl into Chroma 1 (after a server outage):
python -m agents.official_ingestion_pipeline --replay --date 2026-05-03

# Rebuild Chroma 2 schema part if PG ↔ Chroma drift detected:
python -m scripts.rebuild_chroma2_schema --dry-run
python -m scripts.rebuild_chroma2_schema           # applies the rebuild

# Run lifecycle (production hardening Day 4):
python -m scripts.data_admin scan-pending          # find half-finished runs
python -m scripts.data_admin rollback RUN_ID       # delete one run cleanly
python -m scripts.data_admin rollback-all-pending  # crash-recovery sweep
python -m scripts.data_admin show RUN_ID           # print one manifest

# Long-running scheduler (single command bootstraps + then daily cron):
python -m scripts.scheduler                         # daemon
python -m scripts.scheduler --bootstrap             # bootstrap only
python -m scripts.scheduler --once --task official_sources
```

---

## Tests

```bash
pytest tests/                                                 # unit tests
PYTEST_RUN_LIVE_SCHEMA=1 pytest tests/test_schema_consistency.py
                                                              # PG ↔ Chroma 2 live check
```

---

## Evaluation

The `eval/` directory contains a 9-module evaluation suite (530 total test cases).
The API must be running before executing the suite.

```bash
# Run all 9 modules (RAG, NL2SQL, KG, Planner, Report,
#   Echo Chamber, Emotion, Propagation, Claim Verify):
python -m eval.run_eval

# Run specific modules only:
python -m eval.run_eval --modules emotion claim_verify propagation

# Point at a non-default API:
python -m eval.run_eval --base-url http://localhost:8000

# Include the optional LLM-as-judge E2E module (requires GOOGLE_API_KEY):
python -m eval.run_eval --include-e2e
```

Results are written to `eval/results/summary.json`. Open
`eval/results/eval_dashboard.html` in a browser for the full interactive
report — radar chart, per-module metrics with pass/fail badges, and
sample test cases per module.

**Targets (all met in latest run):**

| Module | Key metric | Target | Actual |
|---|---|---|---|
| RAG | Recall@5 | 0.92 | 1.00 |
| NL2SQL | Pass@1 | 0.95 | 0.99 |
| KG | Entity pass rate | 0.80 | 1.00 |
| Planner | Route F1 | 0.87 | 0.993 |
| Report | Critic pass rate | 0.90 | 0.98 |
| Echo Chamber | Classification accuracy | 0.75 | 1.00 |
| Emotion | Non-null rate | 0.80 | 1.00 |
| Propagation | Cascade found rate | 0.60 | 1.00 |
| Claim Verify | Key entity recall | 0.75 | 0.95 |

---

## Configuration knobs

All variables live in `config.py` and can be overridden via `.env`. The
common ones:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | (required) | LLM + embedding access |
| `OPENAI_MODEL` | `gpt-4o` | All LLM calls (rewriter / writer / critic / NL2SQL / topic label) |
| `POSTGRES_DSN` | localhost:5432/society_db | Write connection |
| `POSTGRES_READONLY_DSN` | (falls back to DSN) | NL2SQL connection (recommend separate read-only role in prod) |
| `NL2SQL_MAX_REPAIR_ROUNDS` | 3 | Max self-correction iterations |
| `NL2SQL_RESULT_ROW_LIMIT` | 1000 | Forced LIMIT on every query |
| `NL2SQL_STATEMENT_TIMEOUT_MS` | 5000 | Postgres `statement_timeout` per query |
| `EXPERIENCE_TTL_DAYS` | 30 | Auto-decay age cutoff |
| `KG_TOPIC_SEMANTIC_FALLBACK_MIN_SIM` | 0.30 | Minimum topic similarity for KG fallback when the exact topic has no graph signal |
| `EXPERIENCE_MIN_CONFIDENCE` | 0.2 | Auto-decay confidence floor |
| `MULTIMODAL_DAILY_BUDGET_USD` | 5.0 | Daily image-understanding spend cap |
| `MULTIMODAL_MIN_LIKES / MIN_REPLIES` | 50 / 20 | Sample threshold for image processing |
| `SESSION_MAX_TURNS` | 40 | Per-session conversation window |
| `SESSION_MIN_TURNS_TO_COMPACT` | 10 | Minimum batch size when the compactor runs |
| `SESSION_SUMMARY_MAX_CHARS` | 1200 | Cap on the rolling summary length |

---

## Layout

```
agents/                # Pipeline + chat agents (see workflow.md §4.1)
tools/                 # Atomic operations (hybrid_retrieval / nl2sql / kg / topic_resolver)
services/              # Storage + third-party wrappers
models/                # Pydantic data contracts
api/                   # FastAPI routes (chat, retrieve, import, reflection, runs, artifacts,
                       #   admin/nl2sql, plan)
ui/                    # Single-page Streamlit Chat UI
db/                    # schema_v2.sql
config/                # YAML configs (official_sources.yaml)
scripts/               # CLI utilities (seed, decay, rebuild)
tests/                 # Unit tests + schema consistency
eval/                  # Evaluation suite (9 modules, ground-truth datasets, HTML dashboard)
  eval_rag.py          #   RAG / evidence retrieval
  eval_nl2sql.py       #   NL2SQL SQL generation
  eval_kg.py           #   Knowledge Graph query
  eval_planner.py      #   Query Planner routing
  eval_report.py       #   Report Writer quality
  eval_echo_chamber.py #   Echo Chamber detection
  eval_emotion.py      #   Emotion NL2SQL
  eval_propagation.py  #   Propagation / cascade
  eval_claim_verify.py #   Claim Verification
  ground_truth/        #   Ground-truth JSON datasets for each module
  results/             #   Eval results (eval_dashboard.html — interactive HTML report)
  run_eval.py          #   CLI orchestrator for the full eval suite

main.py                # Backend pipeline CLI
config.py              # Central config (env vars)
PROJECT_REDESIGN_V2.md # Design + decision log
workflow.md            # Architecture + module map (concise)
docs/phase{1..5}_done.md  # Per-phase delivery summaries
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'X' from 'config'` | Stale config var removed during cleanup | Pull latest, no `.env` change needed |
| Chat answer says "I couldn't gather enough data" | Postgres / Chroma empty | Run step 5 (offline ingestion) |
| `branches_used: []` for every query | Planner / Rewriter LLM unreachable | Check `OPENAI_API_KEY`, OpenAI status |
| `needs_human_review: true` on factual answers | Critic flagged a numeric mismatch | Inspect `branch_outputs` — usually means LLM invented a number |
| Reuters / AP returning 0 chunks | Their public RSS is gone; we proxy via Google News | Already handled in `config/official_sources.yaml`; check network |
| `chroma_official: 0` after `--once` | Network blocked (CN region) | Set proxy env or use `--replay` against pre-pulled jsonl |
| Reranker download hangs | First-run model download | Wait for ~600MB; subsequent runs use the local cache |
| Schema-consistency test fails | Chroma 2 drifted from PG | `python -m scripts.rebuild_chroma2_schema` |

---

## Where the design lives

- **`PROJECT_REDESIGN_V2.md`** — full design doc with decision log (Q1-Q11)
- **`workflow.md`** — current code map, module clinic, command cheat sheet
- **`docs/phase{1..5}_done.md`** — what landed in each phase

If you're new to the codebase, read `workflow.md` first; it's the
shortest path from zero to "I know where everything is".
