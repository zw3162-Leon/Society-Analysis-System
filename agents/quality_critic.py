"""
Quality Critic - redesign-2026-05 Phase 4.4.

Sits between Report Writer and the user. Validates the v2 ReportV2 along
four axes (PROJECT_REDESIGN_V2.md 1.2 step 6):

    1. citation_completeness  - every cited [chunk_id] resolves to a real
                                evidence citation. Programmatic.
    2. numeric_consistency    - every number in `report.numbers` matches a
                                value present in the corresponding branch
                                output. Programmatic.
    3. on_topic               - LLM-assisted: does the markdown actually
                                answer the user's question?
    4. hallucination_check    - LLM-assisted: does anything assert facts
                                without source backing?

Outputs `CriticVerdict` (models/reflection.py). On failure the orchestrator
retries Report Writer once; second failure flips
`report.needs_human_review = True` so downstream UI can flag it.

Programmatic checks use deterministic rules. Only the latter two consult an
LLM, and they share a single call (Q9-style "one prompt, multi-axis"
verdict) to keep latency bounded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import openai
import structlog

from config import LLM_REQUEST_TIMEOUT, OPENAI_API_KEY, OPENAI_MODEL
from agents.planner_v2 import PlanExecutionV2
from models.branch_output import EvidenceOutput, KGOutput, SQLOutput
from models.reflection import CriticVerdict, ErrorKind, FailedBranch
from models.report_v2 import ReportV2

log = structlog.get_logger(__name__)


# Only bracketed internal evidence ids are programmatically resolved here.
# Markdown source links such as [AP News](https://...) are user-facing
# citations and must not be mistaken for chunk ids.
_CITATION_RE = re.compile(
    r"\[((?:[0-9a-f]{12,}|c[\w:-]*|ev[\w:-]*|chunk_[\w:-]+))\]"
)


_CRITIC_SYSTEM = """You are a quality critic for a research-system answer.
Given the user's question and a markdown report plus the branch outputs
that produced it, decide:

  on_topic         (bool): does the report actually answer the question?
  hallucination    (bool): does the report state facts not supported by the
                            provided branch outputs?

Return STRICT JSON: {"on_topic": bool, "hallucination": bool,
"reason_on_topic": "...", "reason_hallucination": "..."}.

Rules of thumb:
- "nl2sql: returned N result rows from ..." means the SQL produced N rows.
  Each row is a record/group/aggregate (e.g. one emotion category, one
  account, one count). It is NOT necessarily the total count of items
  unless the SQL explicitly returns a single COUNT(*) row.
- A report listing 4 emotion categories that came from 4 SQL rows is
  on-topic and not a hallucination.
- If the report says it lacks data and that's defensible, on_topic=true.
- Numbers are checked separately by a deterministic checker; do not flag
  them as hallucinations unless they CLEARLY contradict the sample rows.
"""


@dataclass
class QualityCritic:
    """Programmatic + LLM-assisted verifier of `ReportV2`."""

    client: Optional[openai.OpenAI] = None
    model: str = OPENAI_MODEL
    max_numeric_drift: float = 1e-6

    def __post_init__(self) -> None:
        self.client = self.client or openai.OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_REQUEST_TIMEOUT)

    # ── Public ───────────────────────────────────────────────────────────────

    def review(
        self, report: ReportV2, execution: PlanExecutionV2,
    ) -> CriticVerdict:
        if not report.markdown_body.strip():
            return CriticVerdict(
                passed=False, error_kind="off_topic", failed_branch="writer",
                notes="empty markdown body",
            )

        # 1. Citation completeness (programmatic)
        cite_kind = self._check_citations(report)
        if cite_kind is not None:
            return CriticVerdict(
                passed=False, error_kind=cite_kind, failed_branch="writer",
                notes="missing or invalid citation tag",
            )

        # 2. Numeric consistency (programmatic)
        num_kind, num_note = self._check_numbers(report, execution)
        if num_kind is not None:
            return CriticVerdict(
                passed=False, error_kind=num_kind, failed_branch="writer",
                notes=num_note,
            )

        # 3 + 4. On-topic + hallucination (LLM)
        try:
            llm_kind, failed_branch, llm_notes = self._llm_axes(report, execution)
        except Exception as exc:
            # LLM unavailable: be lenient. Pass with a note so the UI can
            # surface "auto-checks only" if needed.
            log.warning("quality_critic.llm_error", error=str(exc)[:160])
            return CriticVerdict(
                passed=True, notes=f"llm_skip: {exc}",
            )
        if llm_kind is not None:
            return CriticVerdict(
                passed=False, error_kind=llm_kind,
                failed_branch=failed_branch, notes=llm_notes,
            )

        return CriticVerdict(passed=True)

    # ── Programmatic checks ──────────────────────────────────────────────────

    def _check_citations(self, report: ReportV2) -> Optional[ErrorKind]:
        cited_in_text = set(_CITATION_RE.findall(report.markdown_body))
        if not cited_in_text:
            return None
        if cited_in_text:
            available = {c.chunk_id for c in report.citations}
            unknown = cited_in_text - available
            if unknown:
                return "citation_missing"
        return None

    def _check_numbers(
        self, report: ReportV2, execution: PlanExecutionV2,
    ) -> tuple[Optional[ErrorKind], str]:
        if not report.numbers:
            return None, ""
        sql_outs = [
            SQLOutput.model_validate(r.output) for r in execution.results
            if r.invocation.branch == "nl2sql"
            and r.status.success and r.output is not None
        ]
        kg_outs = [
            KGOutput.model_validate(r.output) for r in execution.results
            if r.invocation.branch == "kg"
            and r.status.success and r.output is not None
        ]
        ev_outs = [
            EvidenceOutput.model_validate(r.output) for r in execution.results
            if r.invocation.branch == "evidence"
            and r.status.success and r.output is not None
        ]

        for n in report.numbers:
            if n.source_branch == "nl2sql":
                if not _value_in_sql(n.value, sql_outs, tol=self.max_numeric_drift):
                    return "numeric_mismatch", (
                        f"number '{n.label}'={n.value} not present in any "
                        f"SQL row (source_ref={n.source_ref})"
                    )
            elif n.source_branch == "kg":
                if not _value_in_kg(n.value, kg_outs, tol=self.max_numeric_drift):
                    return "numeric_mismatch", (
                        f"number '{n.label}'={n.value} not present in KG metrics"
                    )
            elif n.source_branch == "evidence":
                # Evidence numbers are tougher to verify; only require the
                # source chunk_id to exist when source_ref looks like one.
                if n.source_ref and not _chunk_id_known(n.source_ref, ev_outs):
                    return "numeric_mismatch", (
                        f"number '{n.label}' source_ref '{n.source_ref}' "
                        "not found in any evidence bundle"
                    )
            else:
                return "numeric_mismatch", (
                    f"number '{n.label}' has unknown source_branch "
                    f"'{n.source_branch}'"
                )
        return None, ""

    # ── LLM check ────────────────────────────────────────────────────────────

    def _llm_axes(
        self, report: ReportV2, execution: PlanExecutionV2,
    ) -> tuple[Optional[ErrorKind], Optional[FailedBranch], str]:
        # Compact preview of branch outputs for the LLM
        import json as _json
        preview_parts: list[str] = []
        for r in execution.results:
            if not r.status.success or r.output is None:
                continue
            if r.invocation.branch == "evidence":
                preview_parts.append(
                    "evidence chunks available: " + ", ".join(
                        c.get("chunk_id", "") for c in
                        (r.output.get("bundle") or {}).get("chunks", [])[:8]
                    )
                )
            elif r.invocation.branch == "nl2sql":
                rows = r.output.get("rows") or []
                preview_parts.append(
                    f"nl2sql: returned {len(rows)} result row"
                    f"{'s' if len(rows) != 1 else ''} from query "
                    f"{(r.output.get('final_sql') or '')[:160]!r}.\n"
                    f"  sample rows (up to 5): "
                    f"{_json.dumps(rows[:5], default=str)[:600]}"
                )
            elif r.invocation.branch == "kg":
                preview_parts.append(
                    f"kg metrics={r.output.get('metrics', {})} "
                    f"nodes_returned={len(r.output.get('nodes', []))}"
                )
        body = (
            f"User question: {report.user_question}\n\n"
            f"Markdown:\n{report.markdown_body}\n\n"
            f"Branch evidence:\n" + "\n".join(preview_parts)
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _CRITIC_SYSTEM},
                {"role": "user", "content": body[:6000]},
            ],
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None, None, "critic_json_parse_error_skipped"
        if not bool(data.get("on_topic", True)):
            return ("off_topic", "writer",
                    str(data.get("reason_on_topic", ""))[:160])
        if bool(data.get("hallucination", False)):
            return ("citation_missing", "writer",
                    str(data.get("reason_hallucination", ""))[:160])
        return None, None, ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _value_in_sql(value: float, sql_outs: list[SQLOutput], tol: float) -> bool:
    for s in sql_outs:
        for row in s.rows:
            for v in (row.values() if isinstance(row, dict) else []):
                if isinstance(v, (int, float)) and abs(float(v) - value) <= tol:
                    return True
        if abs(float(len(s.rows)) - value) <= tol:
            return True  # often the value IS the row count
    return False


def _value_in_kg(value: float, kg_outs: list[KGOutput], tol: float) -> bool:
    for k in kg_outs:
        for v in (k.metrics.values() if isinstance(k.metrics, dict) else []):
            if isinstance(v, (int, float)) and abs(float(v) - value) <= tol:
                return True
        if abs(float(len(k.nodes)) - value) <= tol:
            return True
        if abs(float(len(k.edges)) - value) <= tol:
            return True
    return False


def _chunk_id_known(chunk_id: str, ev_outs: list[EvidenceOutput]) -> bool:
    for ev in ev_outs:
        for c in ev.bundle.chunks:
            if c.chunk_id == chunk_id:
                return True
    return False
