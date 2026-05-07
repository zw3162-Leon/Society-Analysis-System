"""Module 9 — Propagation Analysis evaluation.

Tests two KG query kinds:
  - viral_cascade  : top-K reply cascades within a topic (KGQueryTool)
  - propagation_path : reply chain between two accounts (KGQueryTool)

Metrics
-------
api_success_rate        : fraction of API calls returning without error
cascade_found_rate      : viral_cascade entries where cascade_count >= expected_min (≥1)
path_api_valid_rate     : propagation_path entries where response has valid path structure
path_found_accuracy     : propagation_path entries where path_found matches expected
                          (only counted when expected_path_found is not null in GT)
avg_cascade_depth       : mean max_depth across successful viral_cascade results (diagnostic)
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import requests

GT_FILE = Path(__file__).parent / "ground_truth" / "propagation_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "propagation_results.json"


def run(base_url: str) -> dict:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))

    api_success: list[bool] = []
    cascade_found: list[bool] = []
    cascade_depths: list[int] = []
    path_valid: list[bool] = []
    path_accuracy: list[float] = []
    api_errors = 0

    for entry in gt["entries"]:
        qkind = entry["query_kind"]
        target = entry.get("target", {})

        try:
            resp = requests.post(
                f"{base_url}/retrieve/kg",
                json={"query_kind": qkind, "target": target},
                timeout=60,
            )
            resp.raise_for_status()
            kg_output = resp.json().get("kg_output", {})
            api_success.append(True)
        except Exception as exc:
            print(f"  [prop] API error for {entry['id']}: {exc}")
            api_errors += 1
            api_success.append(False)
            continue

        metrics = kg_output.get("metrics", {})
        nodes = kg_output.get("nodes", [])
        edges = kg_output.get("edges", [])

        if qkind == "viral_cascade":
            cascade_count = metrics.get("cascade_count", 0) or len(nodes)
            min_cascades = entry.get("expected_cascade_count_min", 1)
            found = int(cascade_count) >= min_cascades
            cascade_found.append(found)

            max_depth = metrics.get("max_depth", 0) or 0
            if max_depth:
                cascade_depths.append(int(max_depth))

            print(
                f"  [prop] {entry['id']} viral_cascade: "
                f"cascades={cascade_count} depth={max_depth} found={found}"
            )

        elif qkind == "propagation_path":
            # API returns paths_found (int) and max_path_length; nodes always
            # includes source/target stubs even when no path exists.
            paths_found = int(metrics.get("paths_found", 0) or 0)
            path_found = paths_found > 0
            # Validate response structure
            is_valid = isinstance(nodes, list) and isinstance(edges, list)
            path_valid.append(is_valid)

            expected_found = entry.get("expected_path_found")
            if expected_found is not None:
                path_accuracy.append(1.0 if (path_found == expected_found) else 0.0)

            print(
                f"  [prop] {entry['id']} propagation_path: "
                f"paths_found={paths_found} found={path_found} "
                f"expected={expected_found}"
            )

    n = len(gt["entries"])
    n_cascade = sum(1 for e in gt["entries"] if e["query_kind"] == "viral_cascade")
    n_path = sum(1 for e in gt["entries"] if e["query_kind"] == "propagation_path")

    summary = {
        "n_entries": n,
        "n_viral_cascade": n_cascade,
        "n_propagation_path": n_path,
        "api_errors": api_errors,
        "api_success_rate": round(mean(api_success), 4) if api_success else 0.0,
        "cascade_found_rate": round(mean(cascade_found), 4) if cascade_found else 0.0,
        "avg_cascade_depth": round(mean(cascade_depths), 2) if cascade_depths else 0.0,
        "path_api_valid_rate": round(mean(path_valid), 4) if path_valid else 0.0,
        "path_found_accuracy": round(mean(path_accuracy), 4) if path_accuracy else None,
        "n_path_accuracy_tested": len(path_accuracy),
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
