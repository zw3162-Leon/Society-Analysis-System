"""Module 4 — Planner routing evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import requests

GT_FILE = Path(__file__).parent / "ground_truth" / "planner_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "planner_results.json"

ALL_BRANCHES = {"evidence", "nl2sql", "kg"}


def run(base_url: str) -> dict:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    recalls: list[float] = []
    precisions: list[float] = []
    f1s: list[float] = []
    exact_matches: list[bool] = []
    per_branch_recall: dict[str, list] = {b: [] for b in ALL_BRANCHES}
    per_branch_precision: dict[str, list] = {b: [] for b in ALL_BRANCHES}
    api_errors = 0

    for entry in gt["entries"]:
        try:
            resp = requests.post(
                f"{base_url}/plan",
                json={"question": entry["question"],
                      "session_id": f"eval-plan-{entry['id']}"},
                timeout=30,
            )
            resp.raise_for_status()
            predicted = set(resp.json().get("planned_branches", []))
        except Exception as exc:
            print(f"  [planner] API error for {entry['id']}: {exc}")
            api_errors += 1
            continue

        expected = set(entry["expected_branches"])
        intersection = expected & predicted
        recall = len(intersection) / len(expected) if expected else 1.0
        precision = len(intersection) / len(predicted) if predicted else 0.0
        f1 = (2 * recall * precision / (recall + precision)
              if (recall + precision) > 0 else 0.0)
        exact = predicted == expected

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        exact_matches.append(exact)

        for branch in ALL_BRANCHES:
            if branch in expected:
                per_branch_recall[branch].append(1.0 if branch in predicted else 0.0)
            if branch in predicted:
                per_branch_precision[branch].append(1.0 if branch in expected else 0.0)

    summary = {
        "n_queries": len(gt["entries"]),
        "api_errors": api_errors,
        "route_recall": round(mean(recalls), 4) if recalls else 0.0,
        "route_precision": round(mean(precisions), 4) if precisions else 0.0,
        "route_f1": round(mean(f1s), 4) if f1s else 0.0,
        "exact_match_rate": round(mean(exact_matches), 4) if exact_matches else 0.0,
        "per_branch_recall": {
            b: round(mean(v), 4) if v else None
            for b, v in per_branch_recall.items()
        },
        "per_branch_precision": {
            b: round(mean(v), 4) if v else None
            for b, v in per_branch_precision.items()
        },
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
