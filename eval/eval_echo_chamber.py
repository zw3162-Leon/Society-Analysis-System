"""Module 7 — Echo Chamber Detection evaluation.

Tests KGAnalytics.echo_chamber() via POST /retrieve/kg?query_kind=echo_chamber.

Metrics
-------
classification_accuracy  : fraction where is_echo_chamber matches expected label
modularity_in_range_rate : fraction where modularity ∈ [expected_min, expected_max]
community_detection_rate : fraction where community_count >= expected_community_count_min
api_success_rate         : fraction of successful API calls (diagnostic)

Note: classification labels in the GT are domain-knowledge estimates.
Verify against the live system if classification_accuracy appears low.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import requests

GT_FILE = Path(__file__).parent / "ground_truth" / "echo_chamber_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "echo_chamber_results.json"

_SKIP_REASONS = {"graph_too_small", "deps_missing", "calc_error",
                 "networkx_missing", "louvain_missing"}


def run(base_url: str) -> dict:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))

    classification_scores: list[float] = []
    modularity_in_range: list[bool] = []
    community_detected: list[bool] = []
    api_errors = 0
    skipped_small = 0
    by_topic: dict[str, list[float]] = {}

    for entry in gt["entries"]:
        try:
            resp = requests.post(
                f"{base_url}/retrieve/kg",
                json={
                    "query_kind": "echo_chamber",
                    "target": {
                        "topic_id": entry["topic_id"],
                        "modularity_threshold": entry.get("modularity_threshold", 0.3),
                    },
                },
                timeout=60,
            )
            resp.raise_for_status()
            kg_output = resp.json().get("kg_output", {})
            metrics = kg_output.get("metrics", {})
        except Exception as exc:
            print(f"  [echo] API error for {entry['id']}: {exc}")
            api_errors += 1
            continue

        reason = metrics.get("reason", "")
        if reason in _SKIP_REASONS:
            print(f"  [echo] {entry['id']}: skipped ({reason})")
            skipped_small += 1
            continue

        is_ec = metrics.get("is_echo_chamber")
        modularity = metrics.get("modularity", 0.0)
        community_count = metrics.get("community_count", 0)

        expected = entry.get("expected_is_echo_chamber")
        if expected is not None and is_ec is not None:
            score = 1.0 if (is_ec == expected) else 0.0
            classification_scores.append(score)
            by_topic.setdefault(entry.get("topic_label", entry["topic_id"]), []).append(score)

        mod_min = entry.get("expected_modularity_min", 0.0)
        mod_max = entry.get("expected_modularity_max", 1.0)
        if modularity is not None:
            modularity_in_range.append(mod_min <= float(modularity) <= mod_max)

        min_comm = entry.get("expected_community_count_min", 1)
        community_detected.append(int(community_count) >= min_comm)

        print(
            f"  [echo] {entry['id']}: modularity={modularity:.3f} "
            f"is_echo={is_ec} expected={expected} communities={community_count}"
        )

    n = len(gt["entries"])
    api_successes = n - api_errors
    summary = {
        "n_entries": n,
        "api_successes": api_successes,
        "api_errors": api_errors,
        "skipped_small_graph": skipped_small,
        "api_success_rate": round(api_successes / n, 4) if n else 0.0,
        "classification_accuracy": round(mean(classification_scores), 4) if classification_scores else 0.0,
        "modularity_in_range_rate": round(mean(modularity_in_range), 4) if modularity_in_range else 0.0,
        "community_detection_rate": round(mean(community_detected), 4) if community_detected else 0.0,
        "n_classified": len(classification_scores),
        "classification_by_topic": {k: round(mean(v), 4) for k, v in by_topic.items()},
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
