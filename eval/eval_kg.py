"""Module 3 — KG (Knowledge Graph) evaluation using structured /retrieve/kg calls.

Each GT entry specifies a topic_pair = [topic_a, topic_b]. The eval calls
POST /retrieve/kg with query_kind=topic_correlation and checks whether the
expected entity names appear in the returned Entity nodes.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import requests

GT_FILE = Path(__file__).parent / "ground_truth" / "kg_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "kg_results.json"


def run(base_url: str) -> dict:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    entity_recall_all: list[float] = []
    entity_pass_all: list[bool] = []
    by_category: dict[str, list] = {}
    api_errors = 0

    for entry in gt["entries"]:
        topic_pair = entry.get("topic_pair", [])
        if len(topic_pair) != 2:
            print(f"  [kg] skipping {entry['id']}: missing topic_pair")
            api_errors += 1
            continue

        try:
            resp = requests.post(
                f"{base_url}/retrieve/kg",
                json={
                    "query_kind": "topic_correlation",
                    "target": {"topic_a": topic_pair[0], "topic_b": topic_pair[1]},
                },
                timeout=60,
            )
            resp.raise_for_status()
            kg_output = resp.json().get("kg_output", {})
            nodes = kg_output.get("nodes", [])
        except Exception as exc:
            print(f"  [kg] API error for {entry['id']}: {exc}")
            api_errors += 1
            continue

        returned_names = {
            n["properties"].get("name", "").lower()
            for n in nodes
            if n.get("label") == "Entity" and n.get("properties", {}).get("name")
        }

        expected = entry["expected_entities"]
        hits = sum(1 for e in expected if e.lower() in returned_names)
        recall = hits / len(expected) if expected else 0.0
        passed = hits >= entry["expected_min_entity_hits"]

        entity_recall_all.append(recall)
        entity_pass_all.append(passed)

        cat = entry.get("category", "unknown")
        by_category.setdefault(cat, []).append(recall)

    summary = {
        "n_queries": len(gt["entries"]),
        "api_errors": api_errors,
        "entity_recall": round(mean(entity_recall_all), 4) if entity_recall_all else 0.0,
        "entity_pass_rate": round(mean(entity_pass_all), 4) if entity_pass_all else 0.0,
        "entity_recall_by_category": {
            k: round(mean(v), 4) for k, v in by_category.items()
        },
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
