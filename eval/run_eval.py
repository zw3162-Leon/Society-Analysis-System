"""CLI orchestrator for the full eval suite.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --modules rag nl2sql kg planner report
    python -m eval.run_eval --modules echo_chamber emotion propagation claim_verify
    python -m eval.run_eval --include-e2e          # also run the Gemini-judge E2E module
    python -m eval.run_eval --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
from pathlib import Path

BASE_URL_DEFAULT = "http://localhost:8000"

MODULE_MAP = {
    # ── Core retrieval pipeline ──────────────────────────────────────────────
    "rag":           "eval.eval_rag",
    "nl2sql":        "eval.eval_nl2sql",
    "kg":            "eval.eval_kg",
    "planner":       "eval.eval_planner",
    "report":        "eval.eval_report",
    # ── Feature-specific evaluations ────────────────────────────────────────
    "echo_chamber":  "eval.eval_echo_chamber",
    "emotion":       "eval.eval_emotion",
    "propagation":   "eval.eval_propagation",
    "claim_verify":  "eval.eval_claim_verify",
    # ── LLM-as-judge (optional, requires GOOGLE_API_KEY) ────────────────────
    "e2e":           "eval.eval_e2e",
}

# Default modules to run (E2E excluded — requires Gemini API key and is slow).
# Use --include-e2e to add it, or --modules e2e to run it alone.
DEFAULT_MODULES = [
    "rag", "nl2sql", "kg", "planner", "report",
    "echo_chamber", "emotion", "propagation", "claim_verify",
]

TARGETS = {
    "rag":          {"recall_at_5": 0.92, "ndcg_at_10": 0.90, "mrr": 0.85},
    "nl2sql":       {"pass_at_1": 0.95, "result_accuracy": 0.90},
    "kg":           {"entity_recall": 0.75, "entity_pass_rate": 0.80},
    "planner":      {"route_recall": 0.90, "route_f1": 0.87},
    "report":       {"critic_pass_rate": 0.90, "first_pass_rate": 0.75},
    # ── Feature evals ────────────────────────────────────────────────────────
    "echo_chamber": {
        "api_success_rate": 0.95,
        "classification_accuracy": 0.75,
        "modularity_in_range_rate": 0.90,
    },
    "emotion": {
        "sql_pass_rate": 0.95,
        "non_null_rate": 0.80,
        "emotion_in_set_rate": 0.75,
    },
    "propagation": {
        "api_success_rate": 0.95,
        "cascade_found_rate": 0.60,
        "path_api_valid_rate": 0.90,
    },
    "claim_verify": {
        "api_success_rate": 0.90,
        "key_entity_recall": 0.75,
        "evidence_found_rate": 0.80,
    },
    # ── LLM-as-judge (optional) ───────────────────────────────────────────────
    "e2e":          {"answer_relevance": 0.80},
}


def _check_targets(module: str, result: dict) -> list[str]:
    failures = []
    for metric, target in TARGETS.get(module, {}).items():
        val = result.get(metric)
        if val is not None and isinstance(val, (int, float)) and val < target:
            failures.append(f"{metric}={val:.4f} < target {target}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run society-analysis-system eval suite")
    parser.add_argument(
        "--modules", nargs="*",
        default=None,
        choices=list(MODULE_MAP.keys()),
        help="Which modules to run (default: all except e2e)",
    )
    parser.add_argument(
        "--include-e2e", action="store_true",
        help="Also run the LLM-as-judge E2E module (requires GOOGLE_API_KEY)",
    )
    parser.add_argument(
        "--base-url", default=BASE_URL_DEFAULT,
        help=f"API base URL (default: {BASE_URL_DEFAULT})",
    )
    args = parser.parse_args()

    if args.modules is not None:
        modules_to_run = args.modules
    else:
        modules_to_run = list(DEFAULT_MODULES)
        if args.include_e2e:
            modules_to_run.append("e2e")

    results: dict = {}
    all_failures: dict[str, list[str]] = {}

    for name in modules_to_run:
        print(f"\n{'='*55}")
        print(f"Running module: {name.upper()}")
        print(f"{'='*55}")
        mod = importlib.import_module(MODULE_MAP[name])
        result = mod.run(args.base_url)
        results[name] = result

        failures = _check_targets(name, result)
        if failures:
            all_failures[name] = failures
            print(f"  TARGETS MISSED: {', '.join(failures)}")
        else:
            print(f"  All targets met.")

        for k, v in result.items():
            if isinstance(v, (int, float)):
                target = TARGETS.get(name, {}).get(k)
                mark = ""
                if target is not None:
                    mark = " PASS" if v >= target else " FAIL"
                print(f"    {k}: {v}{mark}")

    summary = {
        "run_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": args.base_url,
        "modules_run": modules_to_run,
        "modules": results,
    }

    out = Path(__file__).parent / "results" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"Summary written to {out}")
    if all_failures:
        print("\nFAILED TARGETS:")
        for mod_name, fails in all_failures.items():
            for f in fails:
                print(f"  [{mod_name}] {f}")
    else:
        print("\nAll targets met across all modules.")


if __name__ == "__main__":
    main()
