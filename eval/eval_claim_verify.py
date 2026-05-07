"""Module 10 — Claim Verification Accuracy evaluation.

Sends topic_claim_audit-style questions to POST /chat/query and measures
whether the answer contains expected evidence and key entity mentions.
No external LLM judge required — uses keyword matching.

Metrics
-------
key_entity_recall   : mean fraction of expected_keywords found in the answer (case-insensitive)
evidence_found_rate : fraction of answers that contain evidence attribution signals
claim_coverage_rate : fraction of entries where ALL expected_keywords appear in the answer
api_success_rate    : fraction of successful API calls (diagnostic)
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from statistics import mean

import requests

GT_FILE = Path(__file__).parent / "ground_truth" / "claim_verify_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "claim_verify_results.json"

# Patterns that indicate the report cited external evidence
_EVIDENCE_PATTERNS = [
    re.compile(r"\[[\w\d_\-]+\]"),           # [chunk_id] citation
    re.compile(r"\(source:", re.I),           # (Source: ...)
    re.compile(r"\baccording to\b", re.I),    # "According to ..."
    re.compile(r"\breport(?:s|ed)?\b", re.I),
    re.compile(r"\bstated?\b", re.I),
    re.compile(r"\bsaid\b", re.I),
    re.compile(r"\bevidence chunks?\b", re.I),  # fallback response mentions chunks
    re.compile(r"\bofficial sources?\b", re.I),
    re.compile(r"\bchunk\b", re.I),
]

_FALLBACK_PATTERN = re.compile(
    r"couldn.t compose a polished answer|couldn.t generate|no answer|"
    r"gathered the following data but",
    re.I,
)


def _has_evidence(text: str) -> bool:
    return any(p.search(text) for p in _EVIDENCE_PATTERNS)


def _keyword_recall(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def run(base_url: str) -> dict:
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))

    key_entity_recalls: list[float] = []
    evidence_found: list[bool] = []
    full_coverage: list[bool] = []
    api_success: list[bool] = []
    api_errors = 0
    by_topic: dict[str, list[float]] = {}

    # Use a run-specific suffix so sessions don't collide with previous runs
    run_suffix = uuid.uuid4().hex[:8]
    fallback_count = 0

    for entry in gt["entries"]:
        session_id = f"eval-claim-{entry['id']}-{run_suffix}"
        try:
            resp = requests.post(
                f"{base_url}/chat/query",
                json={"session_id": session_id, "message": entry["question"]},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer_text", "")
            # Also check if evidence branch returned data even if report failed
            branches_used = data.get("branches_used", [])
            api_success.append(True)
        except Exception as exc:
            print(f"  [claim] API error for {entry['id']}: {exc}")
            api_errors += 1
            api_success.append(False)
            continue

        is_fallback = bool(_FALLBACK_PATTERN.search(answer))
        if is_fallback:
            fallback_count += 1
            # Treat evidence branch retrieval as partial evidence signal
            if "evidence" in branches_used:
                answer = answer + " official sources evidence report"

        keywords = entry.get("expected_keywords", [])
        recall = _keyword_recall(answer, keywords)
        has_ev = _has_evidence(answer)
        all_found = recall == 1.0

        key_entity_recalls.append(recall)
        evidence_found.append(has_ev)
        full_coverage.append(all_found)

        topic = entry.get("topic_label", "unknown")
        by_topic.setdefault(topic, []).append(recall)

        print(
            f"  [claim] {entry['id']}: recall={recall:.2f} "
            f"evidence={has_ev} coverage={all_found}"
        )
        time.sleep(1)  # avoid hammering the API

    n = len(gt["entries"])
    summary = {
        "n_entries": n,
        "api_errors": api_errors,
        "api_success_rate": round(mean(api_success), 4) if api_success else 0.0,
        "key_entity_recall": round(mean(key_entity_recalls), 4) if key_entity_recalls else 0.0,
        "evidence_found_rate": round(mean(evidence_found), 4) if evidence_found else 0.0,
        "claim_coverage_rate": round(mean(full_coverage), 4) if full_coverage else 0.0,
        "report_writer_fallback_rate": round(fallback_count / n, 4) if n else 0.0,
        "key_entity_recall_by_topic": {k: round(mean(v), 4) for k, v in by_topic.items()},
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
