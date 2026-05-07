"""Module 2 — NL2SQL evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import requests

from eval.utils import check_expected_value

GT_FILE = Path(__file__).parent / "ground_truth" / "nl2sql_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "nl2sql_results.json"


def run(base_url: str) -> dict:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    pass_at_1_all: list[bool] = []
    exec_acc_all: list[bool] = []
    result_acc_all: list[bool] = []
    by_difficulty: dict[str, list] = {}
    api_errors = 0

    for case in gt["cases"]:
        try:
            resp = requests.post(
                f"{base_url}/retrieve/nl2sql",
                json={"nl_query": case["nl_query"]},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json().get("sql_output", {})
        except Exception as exc:
            print(f"  [nl2sql] API error for {case['id']}: {exc}")
            api_errors += 1
            continue

        success = data.get("success", False)
        rows = data.get("rows")

        p1 = bool(success)
        ea = bool(success) and rows is not None
        ra = False
        if ea:
            n = len(rows)
            in_range = case["expected_rows_min"] <= n <= case["expected_rows_max"]
            value_ok = check_expected_value(rows, case.get("expected_value"))
            ra = in_range and value_ok

        pass_at_1_all.append(p1)
        exec_acc_all.append(ea)
        result_acc_all.append(ra)

        diff = case.get("difficulty", "unknown")
        by_difficulty.setdefault(diff, []).append(p1)

    summary = {
        "n_cases": len(gt["cases"]),
        "api_errors": api_errors,
        "pass_at_1": round(mean(pass_at_1_all), 4) if pass_at_1_all else 0.0,
        "execution_accuracy": round(mean(exec_acc_all), 4) if exec_acc_all else 0.0,
        "result_accuracy": round(mean(result_acc_all), 4) if result_acc_all else 0.0,
        "pass_at_1_by_difficulty": {
            k: round(mean(v), 4) for k, v in by_difficulty.items()
        },
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
