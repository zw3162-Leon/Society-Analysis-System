"""Module 1 — RAG (Evidence Retrieval) evaluation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

import requests

from eval.utils import ndcg_at_k, recall_at_k, reciprocal_rank

GT_FILE = Path(__file__).parent / "ground_truth" / "rag_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "rag_results.json"


def run(base_url: str, skip_citations: bool = True) -> dict:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    metric_keys = ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
                   "ndcg_at_5", "ndcg_at_10", "mrr"]
    metrics: dict[str, list] = {k: [] for k in metric_keys}
    by_source: dict[str, list] = {}
    by_type: dict[str, list] = {}
    citations: list[int] = []
    api_errors = 0

    for entry in gt["entries"]:
        try:
            resp = requests.post(
                f"{base_url}/retrieve/evidence",
                json={"query": entry["query"], "top_k": 10},
                timeout=30,
            )
            resp.raise_for_status()
            chunks = resp.json().get("bundle", {}).get("chunks", [])
            retrieved_ids = [c["chunk_id"] for c in chunks]
        except Exception as exc:
            print(f"  [rag] API error for {entry['id']}: {exc}")
            api_errors += 1
            continue

        relevant = set(entry["relevant_chunk_ids"])
        metrics["recall_at_1"].append(recall_at_k(retrieved_ids, relevant, 1))
        metrics["recall_at_3"].append(recall_at_k(retrieved_ids, relevant, 3))
        metrics["recall_at_5"].append(recall_at_k(retrieved_ids, relevant, 5))
        metrics["recall_at_10"].append(recall_at_k(retrieved_ids, relevant, 10))
        metrics["ndcg_at_5"].append(ndcg_at_k(retrieved_ids, relevant, 5))
        metrics["ndcg_at_10"].append(ndcg_at_k(retrieved_ids, relevant, 10))
        metrics["mrr"].append(reciprocal_rank(retrieved_ids, relevant))

        src = entry.get("source", "unknown")
        qt = entry.get("query_type", "unknown")
        r5 = recall_at_k(retrieved_ids, relevant, 5)
        by_source.setdefault(src, []).append(r5)
        by_type.setdefault(qt, []).append(r5)

        if not skip_citations:
            try:
                cr = requests.post(
                    f"{base_url}/chat/query",
                    json={"session_id": f"eval-rag-{entry['id']}", "message": entry["query"]},
                    timeout=180,
                )
                cr_data = cr.json()
                cite_list = cr_data.get("citations", [])
                if cite_list:
                    citations.append(len(cite_list))
                else:
                    answer = cr_data.get("answer_text", "")
                    citations.append(len(re.findall(r"\(Source:", answer)))
            except Exception:
                pass

    summary = {
        "n_queries": len(gt["entries"]),
        "api_errors": api_errors,
        **{k: round(mean(v), 4) if v else 0.0 for k, v in metrics.items()},
        "avg_citations_per_answer": round(mean(citations), 2) if citations else 0.0,
        "recall_at_5_by_source": {k: round(mean(v), 4) for k, v in by_source.items()},
        "recall_at_5_by_type": {k: round(mean(v), 4) for k, v in by_type.items()},
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
