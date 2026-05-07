"""Module 8 — Emotion Detection evaluation (via NL2SQL).

Tests that dominant_emotion queries correctly return non-null values
and that the returned emotion is plausible for the topic type.

Metrics
-------
non_null_rate       : fraction of queries returning a non-empty emotion value
emotion_in_set_rate : fraction where returned emotion is in expected_emotion_set
sql_pass_rate       : fraction of successful SQL executions
by_topic_type       : non_null_rate broken down by topic_type (diagnostic)
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import requests

GT_FILE = Path(__file__).parent / "ground_truth" / "emotion_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "emotion_results.json"

_NULL_VALUES = {"", "none", "null", "not specified", "n/a", "unknown"}


def _extract_emotion(rows: list[dict]) -> str | None:
    """Pull dominant_emotion value from the first result row."""
    if not rows:
        return None
    row = rows[0]
    for key in ("dominant_emotion", "emotion", "mode"):
        val = row.get(key)
        if val is not None:
            return str(val).strip()
    # Try any value if column name is different
    vals = [str(v).strip() for v in row.values() if v is not None]
    return vals[0] if vals else None


def run(base_url: str) -> dict:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))

    non_null_scores: list[bool] = []
    in_set_scores: list[bool] = []
    sql_pass_scores: list[bool] = []
    api_errors = 0
    by_type: dict[str, list[bool]] = {}

    for entry in gt["entries"]:
        try:
            resp = requests.post(
                f"{base_url}/retrieve/nl2sql",
                json={"nl_query": entry["nl_query"]},
                timeout=60,
            )
            resp.raise_for_status()
            sql_out = resp.json().get("sql_output", {})
        except Exception as exc:
            print(f"  [emo] API error for {entry['id']}: {exc}")
            api_errors += 1
            continue

        success = sql_out.get("success", False)
        rows = sql_out.get("rows") or []
        sql_pass_scores.append(success)

        emotion = _extract_emotion(rows) if success else None
        is_non_null = (
            emotion is not None
            and emotion.lower() not in _NULL_VALUES
        )
        non_null_scores.append(is_non_null)

        expected_set = {e.lower() for e in entry.get("expected_emotion_set", [])}
        if is_non_null and expected_set:
            in_set = emotion.lower() in expected_set
            in_set_scores.append(in_set)
        elif expected_set:
            in_set_scores.append(False)

        topic_type = entry.get("topic_type", "unknown")
        by_type.setdefault(topic_type, []).append(is_non_null)

        print(
            f"  [emo] {entry['id']}: sql_pass={success} "
            f"emotion='{emotion}' non_null={is_non_null}"
        )

    n = len(gt["entries"])
    summary = {
        "n_entries": n,
        "api_errors": api_errors,
        "sql_pass_rate": round(mean(sql_pass_scores), 4) if sql_pass_scores else 0.0,
        "non_null_rate": round(mean(non_null_scores), 4) if non_null_scores else 0.0,
        "emotion_in_set_rate": round(mean(in_set_scores), 4) if in_set_scores else 0.0,
        "non_null_by_topic_type": {k: round(mean(v), 4) for k, v in by_type.items()},
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
