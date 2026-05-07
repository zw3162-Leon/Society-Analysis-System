"""Generate eval results dashboard as a self-contained HTML file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUT_FILE = RESULTS_DIR / "eval_dashboard.html"

TARGETS = {
    "rag":     {"recall_at_5": 0.92, "ndcg_at_10": 0.90, "mrr": 0.85},
    "nl2sql":  {"pass_at_1": 0.95, "result_accuracy": 0.90},
    "kg":      {"entity_recall": 0.75, "entity_pass_rate": 0.80},
    "planner": {"route_recall": 0.90, "route_f1": 0.87},
    "report":  {"critic_pass_rate": 0.90, "first_pass_rate": 0.75},
    "e2e":     {"answer_relevance": 0.80},
}

MODULE_LABELS = {
    "rag": "RAG (Evidence Retrieval)",
    "nl2sql": "NL2SQL",
    "kg": "Knowledge Graph",
    "planner": "Planner",
    "report": "Report / Critic",
    "e2e": "E2E (Gemini Judge)",
}

METRIC_LABELS = {
    "recall_at_5": "Recall@5",
    "ndcg_at_10": "NDCG@10",
    "mrr": "MRR",
    "pass_at_1": "Pass@1",
    "result_accuracy": "Result Accuracy",
    "entity_recall": "Entity Recall",
    "entity_pass_rate": "Entity Pass Rate",
    "route_recall": "Route Recall",
    "route_f1": "Route F1",
    "critic_pass_rate": "Critic Pass Rate",
    "first_pass_rate": "First Pass Rate",
    "answer_relevance": "Answer Relevance",
}


def load_summary() -> dict:
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        print("summary.json not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def load_e2e_scores() -> list[float]:
    path = RESULTS_DIR / "e2e_results.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("per_question_scores", [])


def load_e2e_detail() -> list[dict]:
    """Try to load per-question e2e scores from results file."""
    path = RESULTS_DIR / "e2e_results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("per_question", [])


def build_html(summary: dict) -> str:
    modules = summary.get("modules", {})
    run_at = summary.get("run_at", "unknown")

    # ── collect bar chart data ──────────────────────────────────────────────
    chart_data: list[dict] = []
    all_pass = True
    for mod, metrics in TARGETS.items():
        mod_data = modules.get(mod, {})
        for metric, target in metrics.items():
            val = mod_data.get(metric)
            if val is None:
                continue
            passed = val >= target
            if not passed:
                all_pass = False
            chart_data.append({
                "module": mod,
                "label": f"{MODULE_LABELS.get(mod, mod)}\n{METRIC_LABELS.get(metric, metric)}",
                "short": METRIC_LABELS.get(metric, metric),
                "module_label": MODULE_LABELS.get(mod, mod),
                "value": round(float(val), 4),
                "target": target,
                "passed": passed,
            })

    # ── collect extra metrics (non-targeted) ────────────────────────────────
    extra: list[dict] = []
    for mod, mod_data in modules.items():
        targeted = set(TARGETS.get(mod, {}).keys())
        for k, v in mod_data.items():
            if k in targeted or not isinstance(v, (int, float)):
                continue
            if k.startswith("n_") or k in ("api_errors", "judge_errors"):
                continue
            extra.append({
                "module": MODULE_LABELS.get(mod, mod),
                "metric": METRIC_LABELS.get(k, k),
                "value": round(float(v), 4),
            })

    # ── E2E per-question scores ──────────────────────────────────────────────
    e2e_mod = modules.get("e2e", {})
    e2e_n = e2e_mod.get("n_questions", 100)

    chart_json = json.dumps(chart_data)
    extra_json = json.dumps(extra)
    status_color = "#16a34a" if all_pass else "#dc2626"
    status_text = "ALL TARGETS MET" if all_pass else "SOME TARGETS MISSED"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Society Analysis System — Eval Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b; }}
  .header {{ background: #1e293b; color: #f8fafc; padding: 24px 32px; }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; }}
  .header p {{ font-size: 0.875rem; color: #94a3b8; margin-top: 4px; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px;
            font-size: 0.75rem; font-weight: 700; color: white;
            background: {status_color}; margin-top: 8px; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
  .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
           padding: 20px; }}
  .card h2 {{ font-size: 1rem; font-weight: 600; color: #475569; margin-bottom: 16px;
              border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
  .metric-row {{ display: flex; align-items: center; margin-bottom: 10px; }}
  .metric-name {{ width: 160px; font-size: 0.85rem; color: #64748b; flex-shrink: 0; }}
  .bar-wrap {{ flex: 1; background: #f1f5f9; border-radius: 4px; height: 20px;
               position: relative; overflow: hidden; }}
  .bar {{ height: 100%; border-radius: 4px; transition: width .4s; }}
  .bar.pass {{ background: #22c55e; }}
  .bar.fail {{ background: #ef4444; }}
  .target-line {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #94a3b8; }}
  .val {{ width: 52px; text-align: right; font-size: 0.82rem; font-weight: 600;
          padding-left: 8px; }}
  .val.pass {{ color: #16a34a; }}
  .val.fail {{ color: #dc2626; }}
  .tgt {{ font-size: 0.75rem; color: #94a3b8; padding-left: 4px; }}
  .summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .summary-table th {{ background: #f1f5f9; padding: 8px 12px; text-align: left;
                       font-weight: 600; color: #475569; }}
  .summary-table td {{ padding: 8px 12px; border-top: 1px solid #f1f5f9; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 0.72rem;
           font-weight: 600; }}
  .pill.pass {{ background: #dcfce7; color: #166534; }}
  .pill.fail {{ background: #fee2e2; color: #991b1b; }}
  .chart-wrap {{ position: relative; height: 260px; }}
  .errors {{ font-size: 0.8rem; color: #94a3b8; margin-top: 8px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Society Analysis System — Evaluation Dashboard</h1>
  <p>Run: {run_at} &nbsp;|&nbsp; 100 test cases per module (6 modules)</p>
  <div class="badge">{status_text}</div>
</div>
<div class="container">

<!-- ── Module metric bars ─────────────────────────────────────────────── -->
<div class="grid" id="module-cards"></div>

<!-- ── E2E distribution chart ────────────────────────────────────────── -->
<div style="margin-top:16px">
  <div class="card">
    <h2>E2E Answer Relevance Distribution (Gemini 2.5 Pro Judge, n={e2e_n})</h2>
    <div class="chart-wrap"><canvas id="e2eChart"></canvas></div>
    <p class="errors">
      api_errors: {e2e_mod.get("api_errors", 0)} &nbsp;|&nbsp;
      judge_errors: {e2e_mod.get("judge_errors", 0)} &nbsp;|&nbsp;
      answer_relevance (mean): <strong>{e2e_mod.get("answer_relevance", 0):.4f}</strong>
      &nbsp; target ≥ 0.80
    </p>
  </div>
</div>

<!-- ── Additional metrics table ──────────────────────────────────────── -->
<div style="margin-top:16px">
  <div class="card">
    <h2>Additional Metrics (informational)</h2>
    <table class="summary-table" id="extra-table">
      <thead><tr><th>Module</th><th>Metric</th><th>Value</th></tr></thead>
      <tbody id="extra-body"></tbody>
    </table>
  </div>
</div>

</div><!-- /container -->

<script>
const chartData = {chart_json};
const extraData = {extra_json};

// ── Group by module ──────────────────────────────────────────────────────
const modules = {{}};
chartData.forEach(d => {{
  if (!modules[d.module]) modules[d.module] = {{ label: d.module_label, metrics: [] }};
  modules[d.module].metrics.push(d);
}});

const grid = document.getElementById('module-cards');
Object.entries(modules).forEach(([mod, {{label, metrics}}]) => {{
  const card = document.createElement('div');
  card.className = 'card';
  let rows = metrics.map(m => {{
    const pct = Math.min(m.value / 1.0 * 100, 100);
    const tpct = m.target * 100;
    const cls = m.passed ? 'pass' : 'fail';
    return `<div class="metric-row">
      <span class="metric-name">${{m.short}}</span>
      <div class="bar-wrap">
        <div class="bar ${{cls}}" style="width:${{pct.toFixed(1)}}%"></div>
        <div class="target-line" style="left:${{tpct.toFixed(1)}}%"></div>
      </div>
      <span class="val ${{cls}}">${{m.value.toFixed(3)}}</span>
      <span class="tgt">/${{m.target}}</span>
    </div>`;
  }}).join('');
  const allPass = metrics.every(m => m.passed);
  card.innerHTML = `<h2>${{label}} <span class="pill ${{allPass?'pass':'fail'}}">${{allPass?'PASS':'FAIL'}}</span></h2>${{rows}}`;
  grid.appendChild(card);
}});

// ── Extra metrics table ──────────────────────────────────────────────────
const tbody = document.getElementById('extra-body');
extraData.forEach(d => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{d.module}}</td><td>${{d.metric}}</td><td>${{d.value}}</td>`;
  tbody.appendChild(tr);
}});

// ── E2E histogram ────────────────────────────────────────────────────────
// Read per-question scores from embedded data (buckets of 0.1)
const e2eModule = chartData.find(d => d.module === 'e2e');
// We'll build a rough histogram from the known mean — placeholder buckets
// The actual per-question scores are embedded below
const perQ = {json.dumps(_load_per_question_scores(modules))};
const buckets = Array(11).fill(0);
perQ.forEach(s => {{
  const i = Math.min(Math.floor(s * 10), 10);
  buckets[i]++;
}});
const labels = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'];
const colors = labels.map((_, i) => i >= 8 ? '#22c55e' : i >= 5 ? '#f59e0b' : '#ef4444');
new Chart(document.getElementById('e2eChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [{{
      label: 'Questions per score bucket',
      data: buckets,
      backgroundColor: colors,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Relevance Score' }} }},
      y: {{ title: {{ display: true, text: 'Count' }}, beginAtZero: true }},
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


def _load_per_question_scores(modules_placeholder) -> list[float]:
    """Load per-question E2E scores from e2e_results.json if available."""
    path = RESULTS_DIR / "e2e_results.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("per_question_scores", [])


def main() -> None:
    summary = load_summary()
    html = build_html(summary)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {OUT_FILE}")


if __name__ == "__main__":
    main()
