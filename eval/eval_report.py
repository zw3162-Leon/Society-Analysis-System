"""Module 5 — Report Quality evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import requests

GT_FILE = Path(__file__).parent / "ground_truth" / "e2e_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "report_results.json"


def run(base_url: str) -> dict:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    first_pass: list[bool] = []
    reflection_triggered: list[bool] = []
    reflection_recovered: list[bool] = []
    branches_subset: list[bool] = []
    answer_lengths: list[int] = []
    needs_review_flags: list[bool] = []
    api_errors = 0

    for q in gt["questions"]:
        try:
            resp = requests.post(
                f"{base_url}/chat/query",
                json={"session_id": q["session_id"], "message": q["question"]},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  [report] API error for {q['id']}: {exc}")
            api_errors += 1
            continue

        attempts = data.get("critic_attempts", 1)
        passed_on = data.get("critic_passed_on")
        needs_review = data.get("needs_human_review", True)

        first_pass.append(passed_on == "first")
        needs_review_flags.append(needs_review)

        if attempts == 2:
            reflection_triggered.append(True)
            reflection_recovered.append(passed_on == "second")

        branches_used = set(data.get("branches_used", []))
        expected_branches = set(q.get("expected_branches", []))
        branches_subset.append(expected_branches.issubset(branches_used))

        answer_lengths.append(len(data.get("answer_text", "")))

    n = len(first_pass)
    critic_pass_rate = mean([not r for r in needs_review_flags]) if needs_review_flags else 0.0
    reflection_rate = len(reflection_triggered) / len(gt["questions"]) if gt["questions"] else 0.0
    reflection_recovery_rate = (
        mean(reflection_recovered) if reflection_recovered else None
    )

    summary = {
        "n_questions": len(gt["questions"]),
        "api_errors": api_errors,
        "critic_pass_rate": round(critic_pass_rate, 4),
        "hallucination_rate": round(1.0 - critic_pass_rate, 4),
        "first_pass_rate": round(mean(first_pass), 4) if first_pass else 0.0,
        "reflection_rate": round(reflection_rate, 4),
        "reflection_recovery_rate": (
            round(reflection_recovery_rate, 4)
            if reflection_recovery_rate is not None else None
        ),
        "branches_subset_rate": round(mean(branches_subset), 4) if branches_subset else 0.0,
        "avg_answer_length": int(mean(answer_lengths)) if answer_lengths else 0,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
