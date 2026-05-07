"""Module 6 — E2E Answer Relevance (LLM-as-Judge via Gemini 2.5 Pro)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import mean

import requests

GT_FILE = Path(__file__).parent / "ground_truth" / "e2e_gt.json"
RESULTS_FILE = Path(__file__).parent / "results" / "e2e_results.json"

JUDGE_MODEL = "gemini-2.5-pro"
RETRY_DELAYS = [15, 30, 60, 120]

_JUDGE_PROMPT = """\
You are an impartial evaluator for a news analysis AI system.

Question: {question}

Reference facts (what a good answer should cover):
{reference_facts}

System answer:
{answer}

Score on one axis (0.0 to 1.0):
- answer_relevance: does the answer directly and completely address the user's question?
  Penalize vague, off-topic, or incomplete responses.

Return STRICT JSON only:
{{"answer_relevance": <float>, "reasoning": "<one sentence>"}}"""


def _judge_answer(client, question: str, reference_facts: list[str], answer: str) -> float:
    prompt = _JUDGE_PROMPT.format(
        question=question,
        reference_facts="\n".join(f"- {f}" for f in reference_facts),
        answer=answer,
    )
    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw.strip())
            return float(scores["answer_relevance"])
        except Exception as exc:
            print(f"    [e2e] judge attempt {attempt + 1} failed: {exc}")
            if attempt < len(RETRY_DELAYS) - 1:
                time.sleep(delay)
    return 0.0


def run(base_url: str) -> dict:
    try:
        import google.genai as genai
    except ImportError:
        raise ImportError(
            "google-genai is required for E2E eval. "
            "Run: pip install google-genai"
        )

    # Load .env from project root if key not already in environment
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k in ("GOOGLE_API_KEY", "GEMINI_API_KEY") and v.strip():
                os.environ.setdefault("GOOGLE_API_KEY", v.strip())

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY not set in environment or .env")

    client = genai.Client(api_key=api_key)

    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    relevance_scores: list[float] = []
    judge_errors = 0
    api_errors = 0

    api_retry_delays = [10, 20, 40]

    for q in gt["questions"]:
        # Step 1: get answer from system (with retry on connection errors)
        answer = None
        for attempt, delay in enumerate([0] + api_retry_delays):
            if delay:
                time.sleep(delay)
            try:
                resp = requests.post(
                    f"{base_url}/chat/query",
                    json={"session_id": q["session_id"], "message": q["question"]},
                    timeout=180,
                )
                resp.raise_for_status()
                answer = resp.json().get("answer_text", "")
                break
            except Exception as exc:
                print(f"  [e2e] API attempt {attempt + 1} failed for {q['id']}: {exc}")
                if attempt == len(api_retry_delays):
                    api_errors += 1
        if answer is None:
            continue
        # Small delay to avoid overwhelming the API
        time.sleep(2)

        # Step 2: judge the answer
        try:
            score = _judge_answer(
                client,
                question=q["question"],
                reference_facts=q.get("reference_facts", []),
                answer=answer,
            )
            relevance_scores.append(score)
            print(f"  [e2e] {q['id']}: relevance={score:.3f}")
        except Exception as exc:
            print(f"  [e2e] judge error for {q['id']}: {exc}")
            judge_errors += 1

    summary = {
        "n_questions": len(gt["questions"]),
        "api_errors": api_errors,
        "judge_errors": judge_errors,
        "answer_relevance": round(mean(relevance_scores), 4) if relevance_scores else 0.0,
        "per_question_scores": relevance_scores,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
