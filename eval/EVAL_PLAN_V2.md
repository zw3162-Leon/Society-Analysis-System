# Evaluation Plan v2 — Society Analysis System

> **Purpose:** Complete, self-contained specification for running evaluation on the society-analysis-system.
> Feed this entire file to Claude in the new project and ask it to implement and run the eval suite.

---

## 0. Quick Reference

| Module | Endpoint | GT File | Samples |
|--------|----------|---------|---------|
| RAG (Evidence Retrieval) | `POST /retrieve/evidence` | `rag_gt.json` | **100** |
| NL2SQL | `POST /retrieve/nl2sql` | `nl2sql_gt.json` | **100** |
| KG (Knowledge Graph) | `POST /retrieve/kg` or `/chat/query` | `kg_gt.json` | **100** |
| Planner (Route Accuracy) | `POST /plan` | `planner_gt.json` | **100** |
| Report Quality | `POST /chat/query` | `e2e_gt.json` | **100** |
| E2E (LLM-as-Judge) | `POST /chat/query` | `e2e_gt.json` | **100** |

Run all modules:
```bash
python -m eval.run_eval
python -m eval.run_eval --modules rag nl2sql kg planner report e2e --base-url http://localhost:8000
```

---

## 1. Project Layout (Eval-Relevant Files)

```
society-analysis-system/
├── eval/
│   ├── run_eval.py           # CLI entry point
│   ├── eval_rag.py           # RAG evaluation
│   ├── eval_nl2sql.py        # NL2SQL evaluation
│   ├── eval_kg.py            # KG evaluation (new)
│   ├── eval_planner.py       # Planner routing evaluation (new)
│   ├── eval_report.py        # Report quality evaluation
│   ├── eval_e2e.py           # LLM-as-judge (Answer_Relevance only)
│   ├── utils.py              # Shared metrics
│   ├── ground_truth/
│   │   ├── rag_gt.json       # 100 RAG queries
│   │   ├── nl2sql_gt.json    # 100 NL2SQL cases
│   │   ├── kg_gt.json        # 100 KG entity-lookup queries (new)
│   │   ├── planner_gt.json   # 100 routing questions (new)
│   │   └── e2e_gt.json       # 100 end-to-end questions
│   └── results/
│       ├── summary.json
│       ├── rag_results.json
│       ├── nl2sql_results.json
│       ├── kg_results.json
│       ├── planner_results.json
│       ├── report_results.json
│       └── e2e_results.json
├── api/routes/
│   ├── plan.py               # POST /plan — lightweight planner-only endpoint (new)
│   ├── retrieve.py           # /retrieve/evidence, /retrieve/nl2sql, /retrieve/kg
│   └── chat.py               # /chat/query
├── agents/
│   ├── planner_v2.py         # Branch router
│   ├── quality_critic.py     # 4-axis report validator
│   └── chat_orchestrator.py  # Retry loop (max 2 attempts)
```

---

## 2. Module 1 — RAG Evaluation

### 2.1 What It Tests

The hybrid retrieval pipeline (`/retrieve/evidence`) that returns ranked evidence chunks from the Chroma vector DB.

### 2.2 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Recall@1 | Hit in top-1 / total queries | ≥ 0.70 |
| Recall@3 | Hit in top-3 / total queries | ≥ 0.88 |
| Recall@5 | Hit in top-5 / total queries | ≥ 0.92 |
| Recall@10 | Hit in top-10 / total queries | ≥ 0.95 |
| NDCG@5 | Discounted cumulative gain at 5 | ≥ 0.85 |
| NDCG@10 | Discounted cumulative gain at 10 | ≥ 0.90 |
| MRR | Mean 1/(rank of first relevant chunk) | ≥ 0.85 |
| Avg_Citations_Per_Answer | Mean citation count in final reports for these queries | diagnostic |
| Recall@5 by source | Per source (BBC/NYT/Reuters/AP/Xinhua) | diagnostic |
| Recall@5 by type | Per type (original/hard/multi_chunk) | diagnostic |

A "hit" = at least one `relevant_chunk_id` appears in the returned chunk list.

`Avg_Citations_Per_Answer` is measured by also calling `/chat/query` for each query and counting `[chunk_id]` citation references in the final report markdown. It reflects how many retrieved chunks the writer actually uses.

### 2.3 Ground Truth Schema — `rag_gt.json`

```json
{
  "version": "2.0",
  "description": "40 original + 35 hard + 25 multi-chunk = 100 total",
  "source_distribution": {"bbc": 30, "nyt": 25, "reuters": 18, "ap": 15, "xinhua": 12},
  "entries": [
    {
      "id": "rag_001",
      "query": "What institution conducted the study on rising cancer rates in young people?",
      "relevant_chunk_ids": ["e89237998657a8c8e..."],
      "source": "bbc",
      "title": "Article headline here",
      "query_type": "original"
    },
    {
      "id": "rag_041",
      "query": "Which report implies that economic pressures are driving youth migration patterns?",
      "relevant_chunk_ids": ["chunk_id_1", "chunk_id_2"],
      "source": "nyt",
      "title": "Article headline here",
      "query_type": "hard"
    },
    {
      "id": "rag_076",
      "query": "What finding is shared across the UN report and the Reuters article on food insecurity?",
      "relevant_chunk_ids": ["chunk_id_a", "chunk_id_b", "chunk_id_c"],
      "source": "reuters",
      "title": "Multiple articles",
      "query_type": "multi_chunk"
    }
  ]
}
```

**Data breakdown (100 total):**
- `original` — 40: factual lookup, named entity, direct quote retrieval
- `hard` — 35: semantic reasoning, implicit topic, paraphrase matching
- `multi_chunk` — 25: require ≥ 2 chunks, cross-article synthesis

**Source distribution:** BBC 30 | NYT 25 | Reuters 18 | AP 15 | Xinhua 12

### 2.4 API Calls

```python
# Retrieval eval
response = requests.post(f"{base_url}/retrieve/evidence",
    json={"query": entry["query"], "top_k": 10}, timeout=30)
retrieved_ids = [c["chunk_id"] for c in response.json().get("chunks", [])]

# Avg_Citations_Per_Answer: call /chat/query and count [chunk_id] patterns in answer
import re
report_response = requests.post(f"{base_url}/chat/query",
    json={"question": entry["query"], "session_id": f"eval-rag-{entry['id']}"}, timeout=120)
answer = report_response.json().get("answer", "")
citation_count = len(re.findall(r'\[[\w\d]+\]', answer))
```

### 2.5 Eval Logic

```python
for entry in gt["entries"]:
    retrieved = call_retrieve(entry["query"])
    relevant = set(entry["relevant_chunk_ids"])

    metrics["recall_at_1"].append(recall_at_k(retrieved, relevant, k=1))
    metrics["recall_at_3"].append(recall_at_k(retrieved, relevant, k=3))
    metrics["recall_at_5"].append(recall_at_k(retrieved, relevant, k=5))
    metrics["recall_at_10"].append(recall_at_k(retrieved, relevant, k=10))
    metrics["ndcg_at_5"].append(ndcg_at_k(retrieved, relevant, k=5))
    metrics["ndcg_at_10"].append(ndcg_at_k(retrieved, relevant, k=10))
    metrics["mrr"].append(reciprocal_rank(retrieved, relevant))
    metrics["citations"].append(count_citations_in_report(entry["query"], entry["id"]))
```

---

## 3. Module 2 — NL2SQL Evaluation

### 3.1 What It Tests

The natural-language-to-SQL pipeline (`/retrieve/nl2sql`).

**DB state:** posts_v2 ≈ 2498, topics_v2 = 3, entities_v2 = 692, distinct authors = 7

### 3.2 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Pass@1 | Tool reported `success=True` | ≥ 0.95 |
| Execution_Accuracy | success=True AND rows is not None | ≥ 0.95 |
| Result_Accuracy | success AND row count in [min,max] AND value check passes | ≥ 0.90 |
| Pass@1 by difficulty | Easy / Medium / Hard breakdown | diagnostic |

### 3.3 Ground Truth Schema — `nl2sql_gt.json`

```json
{
  "version": "2.0",
  "description": "35 easy + 40 medium + 25 hard = 100 cases",
  "db_state": {"posts_v2": 2498, "topics_v2": 3, "entities_v2": 692, "distinct_authors": 7},
  "cases": [
    {
      "id": "sql_001",
      "nl_query": "How many posts are there in total?",
      "expected_rows_min": 1,
      "expected_rows_max": 1,
      "expected_value": {"op": ">=", "value": 2490},
      "difficulty": "easy",
      "notes": "COUNT(*) FROM posts_v2 ~ 2498"
    },
    {
      "id": "sql_036",
      "nl_query": "Which authors posted more than 50 times about climate topics?",
      "expected_rows_min": 0,
      "expected_rows_max": 7,
      "expected_value": null,
      "difficulty": "medium",
      "notes": "JOIN posts with topics filter"
    },
    {
      "id": "sql_076",
      "nl_query": "Find the top 3 entities appearing in posts by authors who also posted in the last 7 days, ranked by co-occurrence frequency",
      "expected_rows_min": 1,
      "expected_rows_max": 3,
      "expected_value": null,
      "difficulty": "hard",
      "notes": "Multi-table join with subquery and window function"
    }
  ]
}
```

**Difficulty breakdown:** easy 35 / medium 40 / hard 25

**`expected_value` ops:** `"contains"`, `"="`, `">="`, `"<="`

### 3.4 API Call

```python
response = requests.post(f"{base_url}/retrieve/nl2sql",
    json={"nl_query": case["nl_query"]}, timeout=60)
data = response.json()
success = data.get("success", False)
rows = data.get("rows")
```

### 3.5 Eval Logic

```python
for case in gt["cases"]:
    data = call_nl2sql(case["nl_query"])
    pass_at_1 = data["success"]
    exec_acc = data["success"] and data["rows"] is not None
    result_acc = False
    if exec_acc:
        n = len(data["rows"])
        in_range = case["expected_rows_min"] <= n <= case["expected_rows_max"]
        value_ok = check_expected_value(data["rows"], case["expected_value"])
        result_acc = in_range and value_ok
```

---

## 4. Module 3 — KG Evaluation

### 4.1 What It Tests

Whether the knowledge graph retrieval branch surfaces the correct entities for entity-focused questions.

If a dedicated `/retrieve/kg` endpoint exists, call it directly. Otherwise call `/chat/query` with questions that route exclusively to the KG branch, and evaluate the final answer text.

### 4.2 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Entity_Recall | Mean fraction of `expected_entities` mentioned in the response | ≥ 0.75 |
| Entity_Pass_Rate | Fraction of queries where hits ≥ `expected_min_entity_hits` | ≥ 0.80 |
| Entity_Recall by category | Per-category breakdown | diagnostic |

A mention = case-insensitive substring match of the entity name in the response text.
Per-query score = hits / len(expected_entities). Overall = mean across all 100 queries.

### 4.3 Ground Truth Schema — `kg_gt.json`

```json
{
  "version": "1.0",
  "description": "100 entity-lookup queries across 3 categories",
  "category_distribution": {
    "author_entity": 30,
    "topic_entity": 40,
    "cross_entity": 30
  },
  "entries": [
    {
      "id": "kg_001",
      "question": "Which entities are most connected to climate change discussions?",
      "category": "topic_entity",
      "expected_entities": ["IPCC", "Carbon Emissions", "Paris Agreement"],
      "expected_min_entity_hits": 2
    },
    {
      "id": "kg_002",
      "question": "Who are the most active authors posting about economic inequality?",
      "category": "author_entity",
      "expected_entities": ["Author_X", "Author_Y"],
      "expected_min_entity_hits": 1
    },
    {
      "id": "kg_003",
      "question": "Which entities frequently co-occur with migration policy in the graph?",
      "category": "cross_entity",
      "expected_entities": ["UNHCR", "IOM", "EU Commission"],
      "expected_min_entity_hits": 2
    }
  ]
}
```

**Category breakdown (100 total):**
- `author_entity` — 30: "Who posts most about X?", "Which authors are connected to Y?"
- `topic_entity` — 40: "Which entities are associated with topic Z?", "What organizations relate to X?"
- `cross_entity` — 30: "Which entities frequently co-occur?", "What connects entity A and entity B?"

### 4.4 Eval Logic

```python
for entry in gt["entries"]:
    response_text = call_kg(entry["question"])
    expected = entry["expected_entities"]
    hits = sum(1 for e in expected if e.lower() in response_text.lower())
    recall = hits / len(expected) if expected else 0.0
    passed = hits >= entry["expected_min_entity_hits"]
    metrics["entity_recall"].append(recall)
    metrics["entity_pass_rate"].append(passed)
```

### 4.5 GT Generation Notes

- Query the live Kuzu graph first to find real entity names before writing GT entries
- Every `expected_entities` value must be a node that actually exists in `data/kuzu_graph/`
- Set `expected_min_entity_hits` conservatively (typically 1–2) to avoid brittleness

---

## 5. Module 4 — Planner Routing Evaluation

### 5.1 What It Tests

The planner's (`planner_v2.py`) branch selection decisions: given a question, does it correctly identify which retrieval branches to call?

**Failure modes (by severity):**
1. **Miss / false negative** — should call branch X but doesn't → answer will be incomplete (most severe)
2. **Over-route / false positive** — calls an unnecessary extra branch → wastes compute, answer still correct
3. **Complete misroute** — calls wrong branch entirely → answer will be wrong

### 5.2 New Endpoint Required — `POST /plan`

Add a lightweight endpoint that runs only the planner and returns its decision, without executing any retrieval branches:

```python
# api/routes/plan.py
@router.post("/plan")
async def plan(request: PlanRequest):
    planned_branches = planner.route(request.question, request.session_id)
    return {"planned_branches": planned_branches}

# Request / Response
POST /plan
{"question": "...", "session_id": "..."}
→ {"planned_branches": ["evidence", "nl2sql"]}
```

This keeps planner eval fast and independent of the full pipeline.

### 5.3 Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Route_Recall | mean(\|expected ∩ predicted\| / \|expected\|) | ≥ 0.90 |
| Route_Precision | mean(\|expected ∩ predicted\| / \|predicted\|) | ≥ 0.85 |
| Route_F1 | harmonic mean of recall and precision | ≥ 0.87 |
| Exact_Match_Rate | fraction where predicted == expected exactly | diagnostic |
| Per_Branch_Recall | recall computed per branch (evidence/nl2sql/kg) | diagnostic |
| Per_Branch_Precision | precision computed per branch | diagnostic |

**Primary metric is Route_Recall** — missing a required branch is worse than calling an extra one.

### 5.4 Ground Truth Schema — `planner_gt.json`

```json
{
  "version": "1.0",
  "description": "100 routing questions covering all branch combinations",
  "branch_distribution": {
    "evidence": 20,
    "nl2sql": 20,
    "kg": 15,
    "evidence+nl2sql": 15,
    "evidence+kg": 15,
    "nl2sql+kg": 10,
    "evidence+nl2sql+kg": 5
  },
  "entries": [
    {
      "id": "plan_001",
      "question": "What did Reuters report about food insecurity last month?",
      "expected_branches": ["evidence"],
      "routing_rationale": "Pure article retrieval, no structured data or graph needed"
    },
    {
      "id": "plan_021",
      "question": "How many posts were published about inflation last week?",
      "expected_branches": ["nl2sql"],
      "routing_rationale": "Structured count query on posts table"
    },
    {
      "id": "plan_041",
      "question": "Which entities are most connected to the climate change discussion network?",
      "expected_branches": ["kg"],
      "routing_rationale": "Graph traversal for entity relationships"
    },
    {
      "id": "plan_056",
      "question": "What are the main arguments about immigration in recent articles, and how frequently is this topic posted about?",
      "expected_branches": ["evidence", "nl2sql"],
      "routing_rationale": "Article content (evidence) + post frequency count (nl2sql)"
    },
    {
      "id": "plan_071",
      "question": "What do news articles say about climate change, and which entities are central to this topic in the knowledge graph?",
      "expected_branches": ["evidence", "kg"],
      "routing_rationale": "Article content (evidence) + entity network (kg)"
    },
    {
      "id": "plan_081",
      "question": "Which authors post most about migration, and how are they connected to related entities in the graph?",
      "expected_branches": ["nl2sql", "kg"],
      "routing_rationale": "Author frequency (nl2sql) + entity relationships (kg)"
    },
    {
      "id": "plan_091",
      "question": "Give a comprehensive analysis of climate discourse: news coverage, posting trends, and key entity relationships",
      "expected_branches": ["evidence", "nl2sql", "kg"],
      "routing_rationale": "Requires all three branches for complete analysis"
    }
  ]
}
```

**Branch distribution (100 total):**

| Route | Count | Description |
|-------|-------|-------------|
| evidence only | 20 | Questions about article content, quotes, specific reports |
| nl2sql only | 20 | Questions about counts, frequencies, author stats |
| kg only | 15 | Questions about entity relationships, graph structure |
| evidence + nl2sql | 15 | Content + volume/frequency |
| evidence + kg | 15 | Content + entity network |
| nl2sql + kg | 10 | Structured data + graph |
| all three | 5 | Comprehensive analysis questions |

### 5.5 Eval Logic

```python
for entry in gt["entries"]:
    response = requests.post(f"{base_url}/plan",
        json={"question": entry["question"], "session_id": f"eval-plan-{entry['id']}"},
        timeout=30)
    predicted = set(response.json().get("planned_branches", []))
    expected = set(entry["expected_branches"])

    intersection = expected & predicted
    recall = len(intersection) / len(expected) if expected else 1.0
    precision = len(intersection) / len(predicted) if predicted else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    exact = predicted == expected

    metrics["recall"].append(recall)
    metrics["precision"].append(precision)
    metrics["f1"].append(f1)
    metrics["exact_match"].append(exact)

    # Per-branch tracking
    all_branches = {"evidence", "nl2sql", "kg"}
    for branch in all_branches:
        if branch in expected:
            per_branch_recall[branch].append(1.0 if branch in predicted else 0.0)
        if branch in predicted:
            per_branch_precision[branch].append(1.0 if branch in expected else 0.0)

summary["route_recall"] = mean(metrics["recall"])
summary["route_precision"] = mean(metrics["precision"])
summary["route_f1"] = mean(metrics["f1"])
summary["exact_match_rate"] = mean(metrics["exact_match"])
summary["per_branch_recall"] = {b: mean(v) for b, v in per_branch_recall.items()}
summary["per_branch_precision"] = {b: mean(v) for b, v in per_branch_precision.items()}
```

---

## 6. Module 5 — Report Quality Evaluation

### 6.1 What It Tests

Full pipeline output from `/chat/query`: critic quality, reflection mechanism, branch routing completeness, and answer length.

### 6.2 Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Critic_Pass_Rate | Fraction where `needs_human_review=False` | ≥ 0.90 |
| Hallucination_Rate | Fraction where `needs_human_review=True` | ≤ 0.10 |
| First_Pass_Rate | Fraction passing critic on first attempt without retry | ≥ 0.75 |
| Reflection_Rate | Fraction that triggered a critic retry | diagnostic |
| Reflection_Recovery_Rate | Among retried, fraction that passed on second attempt | ≥ 0.50 |
| Branches_Subset_Rate | All `expected_branches` are a subset of branches actually used | ≥ 0.95 |
| Avg_Answer_Length | Mean length of markdown answer (chars) | diagnostic |

### 6.3 Ground Truth Schema — `e2e_gt.json`

```json
{
  "version": "2.0",
  "questions": [
    {
      "id": "e2e_001",
      "session_id": "s-eval-001",
      "question": "What topics are trending and who is amplifying them?",
      "reference_facts": [
        "Topic clusters should be identified from KG or SQL",
        "Author amplification derived from post frequency"
      ],
      "expected_branches": ["kg", "nl2sql"]
    },
    {
      "id": "e2e_036",
      "session_id": "s-eval-036",
      "question": "What did the BBC report about economic inequality last week?",
      "reference_facts": [
        "Answer should cite specific BBC chunks",
        "Economic inequality framing should be present"
      ],
      "expected_branches": ["evidence"]
    }
  ]
}
```

**Data breakdown (100 total):**
- Evidence-only: ~35 (RAG + citation required)
- NL2SQL-only: ~25 (structured data queries)
- KG-only: ~20 (graph traversal, entity relationships)
- Multi-branch: ~20 (evidence + nl2sql, or evidence + kg)

### 6.4 Reflection Metric — Implementation Checklist

The orchestrator's `_compose_with_critic()` already implements single retry internally. To expose reflection metrics, add two fields to the `/chat/query` response:

```python
# In ChatOrchestrator._compose_with_critic(), return:
{
    "answer": report.markdown,
    "critic_attempts": 1 or 2,
    "critic_passed_on": "first" | "second" | None,  # None = neither attempt passed
    "needs_human_review": report.needs_human_review,
    ...
}
```

Eval computation:

```python
first_pass, reflection_triggered, reflection_recovered = [], [], []

for q in questions:
    result = call_api(q)
    attempts = result.get("critic_attempts", 1)
    passed_on = result.get("critic_passed_on")

    first_pass.append(passed_on == "first")
    if attempts == 2:
        reflection_triggered.append(True)
        reflection_recovered.append(passed_on == "second")

summary["first_pass_rate"] = mean(first_pass)
summary["reflection_rate"] = len(reflection_triggered) / len(questions)
summary["reflection_recovery_rate"] = mean(reflection_recovered) if reflection_recovered else None
```

**Interpretation:**
- `reflection_recovery_rate = None` → reflection never triggered; all reports passed first time (best case)
- `reflection_recovery_rate ≥ 0.50` → retry mechanism is earning its cost
- `reflection_recovery_rate < 0.30` → retry rarely helps; writer or critic needs attention

---

## 7. Module 6 — E2E Answer Relevance (LLM-as-Judge)

### 7.1 What It Tests

Same 100 questions as Module 5. An external LLM judge (Gemini 2.5 Pro) scores whether the answer actually addresses the question.

### 7.2 Metric

| Metric | Range | Target |
|--------|-------|--------|
| Answer_Relevance | 0.0–1.0 | ≥ 0.80 |

Faithfulness scoring is intentionally omitted — the internal critic already checks grounding, and an external judge scoring faithfulness without direct access to raw retrieved chunks produces noisy results.

### 7.3 Judge Prompt

```
You are an impartial evaluator for a news analysis AI system.

Question: {question}

Reference facts (what a good answer should cover):
{reference_facts}

System answer:
{answer}

Score on one axis (0.0 to 1.0):
- answer_relevance: does the answer directly and completely address the user's question?
  Penalize vague, off-topic, or incomplete responses.

Return STRICT JSON only:
{"answer_relevance": <float>, "reasoning": "<one sentence>"}
```

### 7.4 Retry Logic

```python
for attempt in range(4):
    try:
        scores = parse_json(judge_llm.call(prompt))
        break
    except Exception:
        time.sleep([15, 30, 60, 120][attempt])
```

---

## 8. Shared Utilities — `eval/utils.py`

```python
import math, re

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant & set(retrieved[:k]) else 0.0

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if r in relevant else 0.0 for r in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0

def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0

def check_expected_value(rows: list[dict], expected: dict | None) -> bool:
    if expected is None:
        return True
    op, val = expected["op"], expected["value"]
    if not rows:
        return False
    actual = list(rows[0].values())[0]
    if op == "contains":
        return str(val).lower() in str(actual).lower()
    elif op == "=":
        return abs(float(actual) - float(val)) < 1e-6
    elif op == ">=":
        return float(actual) >= float(val)
    elif op == "<=":
        return float(actual) <= float(val)
    return False
```

---

## 9. CLI Orchestrator — `eval/run_eval.py`

```python
MODULE_MAP = {
    "rag":     ("eval.eval_rag",     "rag_gt.json",     "rag_results.json"),
    "nl2sql":  ("eval.eval_nl2sql",  "nl2sql_gt.json",  "nl2sql_results.json"),
    "kg":      ("eval.eval_kg",      "kg_gt.json",      "kg_results.json"),
    "planner": ("eval.eval_planner", "planner_gt.json", "planner_results.json"),
    "report":  ("eval.eval_report",  "e2e_gt.json",     "report_results.json"),
    "e2e":     ("eval.eval_e2e",     "e2e_gt.json",     "e2e_results.json"),
}
```

---

## 10. Summary Output Format — `summary.json`

```json
{
  "run_at": "2026-05-05T12:00:00Z",
  "modules": {
    "rag": {
      "n_queries": 100,
      "api_errors": 0,
      "recall_at_1": 0.0,
      "recall_at_3": 0.0,
      "recall_at_5": 0.0,
      "recall_at_10": 0.0,
      "ndcg_at_5": 0.0,
      "ndcg_at_10": 0.0,
      "mrr": 0.0,
      "avg_citations_per_answer": 0.0,
      "recall_at_5_by_source": {"bbc": 0.0, "nyt": 0.0, "reuters": 0.0, "ap": 0.0, "xinhua": 0.0},
      "recall_at_5_by_type": {"original": 0.0, "hard": 0.0, "multi_chunk": 0.0}
    },
    "nl2sql": {
      "n_cases": 100,
      "api_errors": 0,
      "pass_at_1": 0.0,
      "execution_accuracy": 0.0,
      "result_accuracy": 0.0,
      "pass_at_1_by_difficulty": {"easy": 0.0, "medium": 0.0, "hard": 0.0}
    },
    "kg": {
      "n_queries": 100,
      "api_errors": 0,
      "entity_recall": 0.0,
      "entity_pass_rate": 0.0,
      "entity_recall_by_category": {
        "author_entity": 0.0,
        "topic_entity": 0.0,
        "cross_entity": 0.0
      }
    },
    "planner": {
      "n_queries": 100,
      "api_errors": 0,
      "route_recall": 0.0,
      "route_precision": 0.0,
      "route_f1": 0.0,
      "exact_match_rate": 0.0,
      "per_branch_recall": {"evidence": 0.0, "nl2sql": 0.0, "kg": 0.0},
      "per_branch_precision": {"evidence": 0.0, "nl2sql": 0.0, "kg": 0.0}
    },
    "report": {
      "n_questions": 100,
      "api_errors": 0,
      "critic_pass_rate": 0.0,
      "hallucination_rate": 0.0,
      "first_pass_rate": 0.0,
      "reflection_rate": 0.0,
      "reflection_recovery_rate": null,
      "branches_subset_rate": 0.0,
      "avg_answer_length": 0
    },
    "e2e": {
      "n_questions": 100,
      "judge_errors": 0,
      "answer_relevance": 0.0
    }
  }
}
```

---

## 11. Ground Truth Generation Strategy

### 11.0 Universal Requirements — Apply to ALL GT Files

The following two requirements are **mandatory** for every entry across all six GT files. They take precedence over any module-specific guidance below.

#### Requirement A — Data Alignment: Every question must be grounded in actually imported data

Before writing any GT entry, inspect the live system to confirm that the referenced data exists:

- **RAG GT**: For each query, call `GET /retrieve/evidence` against the live Chroma DB and confirm at least one matching chunk is returned. Record its real `chunk_id` as `relevant_chunk_ids`. Never fabricate chunk IDs or assume an article was ingested — verify it.
- **NL2SQL GT**: For each query, run the expected SQL directly against the live PostgreSQL DB (port 15432) and record the actual row count and values as `expected_rows_min`, `expected_rows_max`, and `expected_value`. Do not estimate row counts; measure them. Every table name, column name, and filter value used in the question must exist in the live schema.
- **KG GT**: For each query, inspect the live Kuzu graph (e.g., via `/retrieve/kg` or a direct Kuzu shell) and confirm every entity listed in `expected_entities` exists as a real node. If a named entity cannot be verified in the graph, remove it from the entry or replace it with one that can be verified.
- **Planner GT and E2E/Report GT**: Every question must reference topics, sources, authors, or time ranges that are present in the imported dataset. Do not invent topics or sources that have no posts in the DB. Before finalising a question, confirm that at least one relevant post, article, or entity for that question exists in the live system.

**Consequence of violation**: A GT entry whose referenced data does not exist in the live system will produce misleading eval results — the module cannot be expected to return correct answers for data it has never seen. Any such entry must be replaced before running eval.

#### Requirement B — Routing Guarantee: Every module-specific question must demonstrably route to its target module

For GT files that test a specific functional module (RAG/evidence, NL2SQL, KG), each question must be formulated such that the planner will route it to the target module. This is not optional: a question that routes to the wrong module produces a false negative that corrupts metrics.

**Validation procedure** (run for every new GT entry before committing it):

```python
# Call the /plan endpoint and check that target_branch appears in predicted_branches
response = requests.post(f"{base_url}/plan",
    json={"question": entry["question"], "session_id": "gt-validation"},
    timeout=30)
predicted = response.json().get("planned_branches", [])
assert target_branch in predicted, (
    f"Entry {entry['id']} does not route to '{target_branch}'. "
    f"Predicted: {predicted}. Rewrite the question."
)
```

**Routing signals by module — use these patterns to guarantee correct routing:**

| Target Module | Question must contain signals like… | Must NOT contain signals like… |
|---------------|--------------------------------------|-------------------------------|
| `evidence` (RAG) | "What did [source] report/say/write about…", "According to [news outlet]…", "Which article…", "Find evidence that…" | "How many", "Count", "Which author posted most", "Who are connected in the graph" |
| `nl2sql` | "How many posts…", "Count the number of…", "Which authors posted most…", "What is the total…", "List all [entity] where [condition]" | "What did [news outlet] report", "Which entities are connected", "Graph structure" |
| `kg` | "Which entities are connected to…", "How are [A] and [B] related in the network…", "Who amplified…", "What is the propagation path of…", "Echo chamber", "Community structure" | "How many posts", "What did BBC say", "Count" |

If calling `POST /plan` on a candidate question does not include the target branch, rewrite the question using the signals above until routing is confirmed. Log which questions required rewrites in a `gt_validation_log.json` file alongside the GT files.

---

### RAG GT (100 entries)
1. Source proportions: BBC 30, NYT 25, Reuters 18, AP 15, Xinhua 12
2. **Original (40):** Named entities, statistics, institutions, dates — answer explicitly in article text
3. **Hard (35):** Paraphrase, inference, implicit topic — "Which report implies...", "What underlying assumption..."
4. **Multi-chunk (25):** Synthesis across ≥ 2 chunks — "Compare what BBC and Reuters say about...", "What finding is shared..."
5. Get real chunk IDs by calling `/retrieve/evidence` against the live Chroma DB

### NL2SQL GT (100 cases)
1. **Easy (35):** `COUNT(*)`, `SELECT WHERE`, single table, `ORDER BY LIMIT`
2. **Medium (40):** 2-table `JOIN`, `GROUP BY + COUNT`, date arithmetic, `IN` subqueries
3. **Hard (25):** 3-table joins, window functions (`ROW_NUMBER`, `RANK`), CTEs, complex `HAVING`
4. Verify all `expected_rows_min/max` by running SQL against the live DB before writing GT

### KG GT (100 entries)
1. Query the live Kuzu graph to find real entity names before writing GT
2. Every `expected_entities` value must be a node that actually exists in `data/kuzu_graph/`
3. Set `expected_min_entity_hits` conservatively (1–2) to avoid brittleness
4. Distribute: author-entity 30, topic-entity 40, cross-entity 30

### Planner GT (100 entries)
1. Design questions that unambiguously require specific branch combinations
2. Single-branch questions should clearly NOT require other branches
3. Multi-branch questions should genuinely need all listed branches to answer completely
4. Distribution: evidence 20, nl2sql 20, kg 15, evidence+nl2sql 15, evidence+kg 15, nl2sql+kg 10, all-three 5

### E2E/Report GT (100 questions)
1. **Evidence-only (~35):** Answers in article text → `expected_branches: ["evidence"]`
2. **NL2SQL-only (~25):** Structured data → `expected_branches: ["nl2sql"]`
3. **KG-only (~20):** Graph queries → `expected_branches: ["kg"]`
4. **Multi-branch (~20):** Mixed → e.g. `expected_branches: ["evidence", "nl2sql"]`

---

## 12. Targets Summary

| Module | Metric | v1 Result | v2 Target |
|--------|--------|-----------|-----------|
| RAG | Recall@5 | 0.958 (n=60) | ≥ 0.92 (n=100) |
| RAG | NDCG@10 | 0.906 | ≥ 0.90 |
| RAG | MRR | 0.896 | ≥ 0.85 |
| NL2SQL | Pass@1 | 1.000 (n=45) | ≥ 0.95 (n=100) |
| NL2SQL | Result_Accuracy | 0.933 | ≥ 0.90 |
| KG | Entity_Recall | — (new) | ≥ 0.75 |
| KG | Entity_Pass_Rate | — (new) | ≥ 0.80 |
| Planner | Route_Recall | — (new) | ≥ 0.90 |
| Planner | Route_F1 | — (new) | ≥ 0.87 |
| Report | Critic_Pass_Rate | 0.933 (n=30) | ≥ 0.90 (n=100) |
| Report | First_Pass_Rate | — (new) | ≥ 0.75 |
| Report | Reflection_Recovery_Rate | — (new) | ≥ 0.50 |
| E2E | Answer_Relevance | 0.825 (n=30) | ≥ 0.80 (n=100) |

---

## 13. Instructions for Claude (New Project)

1. Verify `eval/` directory exists with all 6 eval modules and `ground_truth/` subfolder
2. **Add `POST /plan` endpoint** in `api/routes/plan.py` — call only the planner, return `planned_branches`, no retrieval execution (section 5.2)
3. **Add `critic_attempts` and `critic_passed_on`** to `/chat/query` response (section 6.4 checklist)
4. Check GT files — if missing or below 100 entries, generate them using the schemas in sections 2.3, 3.3, 4.3, 5.4, 6.3 and the strategy in section 11. **Before writing any entry, apply the two universal requirements in section 11.0**: (a) verify every referenced entity/chunk/row exists in the live system, and (b) call `POST /plan` on every module-specific question to confirm it routes to the intended target branch. Replace any entry that fails either check. Record all validation outcomes in `eval/ground_truth/gt_validation_log.json`.
5. Ensure `eval/utils.py` has all functions from section 8
6. Ensure `eval/run_eval.py` has the 6-module map from section 9
7. For KG eval: confirm whether `/retrieve/kg` endpoint exists; if not, use `/chat/query` with questions designed to route exclusively to KG
8. Start the API server (`uvicorn api.main:app --reload` or equivalent)
9. Run: `python -m eval.run_eval`
10. Report results using the format from section 10, comparing against targets in section 12
