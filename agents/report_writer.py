"""
Report Writer - redesign-2026-05 Phase 4.3.

Replaces v1's `services/answer_composer.py` for the v2 chat path. Composes
a markdown report from the aggregated branch outputs (evidence / nl2sql /
kg) produced by `agents/planner_v2.py`.

Design constraints (PROJECT_REDESIGN_V2.md 1.2):
- Only ONE LLM call per invocation. The system prompt forbids inventing
  facts: every sentence with a number must cite a row from `nl2sql.rows`
  or a chunk_id from the evidence bundle.
- Outputs `ReportV2` (markdown_body + citations + numbers). Numbers are
  duplicated to a structured table so the Quality Critic can verify them
  programmatically.
- If the planner failed to produce any branch output, we degrade to a
  short "I couldn't gather enough data" fallback rather than hallucinating.
- Post-processes the markdown to swap raw `topic_*` / `ent_*` ids for
  human-readable labels/names by looking up topics_v2 / entities_v2, except
  when the user explicitly requested raw ids.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import openai
import structlog

from config import LLM_REQUEST_TIMEOUT, OPENAI_API_KEY, OPENAI_MODEL
from models.branch_output import EvidenceOutput, KGOutput, SQLOutput
from models.evidence import Citation
from models.query import RewrittenQuery
from models.report_v2 import ReportNumber, ReportV2
from agents.planner_v2 import PlanExecutionV2

log = structlog.get_logger(__name__)


_TOPIC_ID_RE = re.compile(r"\btopic_[0-9a-f]{6,}\b")
_ENTITY_ID_RE = re.compile(r"\bent_[0-9a-f]{6,}\b")


_WRITER_SYSTEM = """You compose a concise markdown answer for a research user.
You will receive structured outputs from up to three retrieval branches:
  - evidence:  authoritative-source chunks with chunk_ids and citations
  - nl2sql:    SQL rows from the community Postgres store
  - kg:        graph nodes / edges / metrics from Kuzu

Your output JSON shape:
{
  "markdown_body": "...",
  "numbers": [
    {"label": "post_count", "value": 42, "source_branch": "nl2sql",
     "source_ref": "<sql snippet or column>"},
    ...
  ]
}

Hard rules:
- Every numerical claim in the markdown MUST also appear in the numbers
  array, with source_branch matching the actual source.
- Cite evidence chunks inline as `[chunk_id]` whenever you reference a
  fact attributed to an official source.
- Do not expose raw chunk_ids as the only citation in user-facing prose. When
  citing evidence, write a findable source reference with source, title, date
  or URL when available. Example:
  "(Evidence: AP News, \"Mexico City is sinking so quickly...\", 2026-05-01,
  https://apnews.com)".
- Never write placeholder citations such as "[chunk_id not available]" or
  "[citation needed]". If no specific chunk_id supports the claim, classify
  it as insufficient evidence.
- If the user asks something the data doesn't cover, say so explicitly.
- Do NOT invent SQL rows, account ids, or chunk ids.
- Never treat a SQL string as evidence. SQL only tells you what was queried;
  only `rows` contain Reddit/community facts.
- If an SQL output says rows=0, state that no matching Reddit/community rows
  were retrieved. Do not describe Reddit posts, comments, authors, counts, or
  discussion themes for that SQL output.
- For topic-level sentiment, prefer an explicit topic/table aggregate
  (`topics_v2.dominant_emotion` or a `dominant_emotion` SQL column) over
  inferring "Not specified" from missing post-level annotations. If the SQL
  row contains both a topic-level emotion and post-level emotions are missing,
  report the topic-level emotion and mention the post-level limitation only
  if it matters.
- If the user explicitly asks for topic_id or entity_id, preserve those raw
  ids in the markdown and put the human-readable label/name next to them.
- If an SQL row is shown in the payload, use its actual values. Do not write
  "data not shown" or "not fully shown" for columns present in that row.
- For list/table questions backed by SQL rows, preserve the SQL row order and
  do not skip rows before the output limit. If you summarize only some rows,
  use the first N rows in order.
- For SQL list questions where the payload shows 20 or fewer rows, include
  every shown row unless the user explicitly asks for a summary. If you show
  only a subset, say "top N" and do not imply it is the full result.
- For SQL topic lists, use compact one-line bullets:
  "**label** (`topic_id`) - post_count: N; dominant_emotion: X".
  This is still a report, not a table, and it keeps all rows visible.
- Keep the report under 1200 words. Use bullet points for lists.
- Output STRICT JSON only. No prose. No code fences.

Graph (KG) presentation rules:
- KG node ids look like `rc_b1`, `topic_xxx`, `ent_xxx` etc. They are
  internal identifiers - DO NOT show them in the markdown.
- For Post nodes use the `author` and `text` properties. Quote the text
  excerpt directly (truncated as provided) and credit the `author`.
  Format: "_<author>_: \"<text>\"" with the body in quotes.
- For Account nodes use the id ONLY when the user explicitly mentioned
  that handle (e.g. "u_alice"). Otherwise prefer the username/handle if
  one is in `properties`.
- For reply chains, write a flowing sentence like "alice's post about X
  was replied to by bob (\"...\"), then by carol (\"...\")" rather than
  a bare list of post ids.
- For PageRank / Louvain / centrality: name the people, summarise their
  scores in plain language ("most influential", "central bridge",
  "tight cluster of N accounts"), and quote one or two characteristic
  posts when post-level data is in the bundle.
- For Entity nodes (label="Entity"): when the payload lists
  `key_entities_in_graph`, explicitly name those entities in the answer as
  the relevant entities identified in the knowledge graph for this topic.
  For example: "The knowledge graph identifies the following entities as
  relevant to this topic: Iran, IRGC, Strait of Hormuz." Always include
  this sentence when key_entities_in_graph is present in the payload.
- If a KG output has no nodes, no edges, and no positive graph metrics, say
  the graph data is insufficient for that KG analysis. Never turn missing
  graph data into a negative conclusion. In particular, do not say a topic
  "does not form an echo chamber" unless the KG output contains enough graph
  metrics to support that verdict.

Fact-check / official-source verification rules:
- Start with a short verdict using one of these labels:
  "Supported by official sources", "Contradicted by official sources",
  "Not found in official sources", or "Insufficient evidence".
- Official-source verification must be based only on evidence chunks. If no
  evidence chunks were retrieved, say no official-source evidence was retrieved.
- Reddit/community corroboration must be based only on non-empty SQL rows. If
  SQL rows are empty, explicitly say no matching Reddit rows were retrieved.
- If official chunks are related but do not address the exact claim, classify
  the official verification as "Not found in official sources" or
  "Insufficient evidence", not as supported.

Comparison / official-recap rules (ONLY apply in comparison or recap mode):
- When the user asks what official media / official sources say about a
  topic, or asks to compare official coverage against community discussion,
  structure the answer with two sections:
  1. "### Official Media Recap" - summarize what the evidence chunks report,
     citing each source inline (Source: outlet, "title", date, URL).
  2. "### Community Perspective" - summarize what the SQL rows show about
     Reddit/community discussion (volume, dominant emotion, notable authors).
- Do NOT apply the fact-check verdict labels in this mode. Recap and
  comparison are descriptive, not adjudicative; pick neither
  "Supported by official sources" nor "Insufficient evidence".
- If one side has no data (no evidence chunks OR SQL rows=0), still emit the
  corresponding section heading but state explicitly that no data was
  retrieved for that side; do not fabricate it from the other side.

Topic claim audit rules:
- Use this mode when the user asks which claims inside a topic agree with
  official/evidence sources, conflict with them, or lack enough evidence.
- The NL2SQL branch returns rows from the `claims_v2` table directly:
  each row carries `claim_text` (canonical extracted claim, NOT a post body),
  `source_url` (the original article link, when known), `topic_label`,
  `distinct_authors`, and `post_count`. Use `claim_text` verbatim — do NOT
  paraphrase it from a post body, and do NOT invent claim phrasing from
  evidence chunks. The 'Reddit claim:' line MUST quote `claim_text`.
- If `source_url` is present on a claim row, render it inline as a Markdown
  link next to the claim_text so users can open the originating article.
- "Reddit author" attribution: cite `distinct_authors` as a count (e.g.
  "discussed by 12 distinct authors"). Individual usernames are no longer
  passed in the row set; do not invent them.
- Ignore claims with `confidence` < 0.4 unless every other claim is also low.
- Deduplicate near-identical claims by claim_id (rows are already deduped
  by simhash, but identical claim_ids must collapse if you see them twice).
- Report at most 8 claims. If there are many weak insufficient-evidence
  rows, summarize that pattern instead of listing every row.
- Classify each claim with exactly one label:
  "Consistent with official/evidence sources",
  "Contradicted by official/evidence sources", or
  "Insufficient evidence".
- Consistent/Contradicted claims must cite at least one evidence chunk inline.
  If no chunk directly addresses the claim, use "Insufficient evidence".
- If an evidence chunk supports the same broader factual assertion as a
  Reddit row, classify that row as consistent even when wording differs.
- For every Consistent/Contradicted claim, include both:
  "Reddit claim: ..." and "Evidence says: ...". The evidence sentence must
  summarize the official/evidence source's actual wording, not just say that
  it supports or contradicts the claim.
- Do not infer a citation from a Reddit row. Reddit rows identify claims and
  authors; evidence chunks are the only source for official/evidence citations.
- Do not use a deterministic table. Write a report with grouped bullets:
  consistent claims, contradicted claims, insufficient-evidence claims.
"""


class ReportWriter:
    def __init__(
        self,
        client: Optional[openai.OpenAI] = None,
        model: str = OPENAI_MODEL,
    ) -> None:
        self._client = client or openai.OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_REQUEST_TIMEOUT)
        self._model = model

    # ── Public ─────────────────────────────────────────────────────────────────

    def write(
        self,
        rq: RewrittenQuery,
        execution: PlanExecutionV2,
    ) -> ReportV2:
        report = ReportV2(
            user_question=rq.original,
            branches_used=list(execution.branches_used),
        )

        # Aggregate outputs by branch
        evidence_outs = _filter_outputs(execution, "evidence", EvidenceOutput)
        sql_outs = _filter_outputs(execution, "nl2sql", SQLOutput)
        kg_outs = _filter_outputs(execution, "kg", KGOutput)
        failed = _failed_results(execution)

        # Hoist all citations even if the LLM omits them in prose
        for ev in evidence_outs:
            for chunk in ev.bundle.chunks:
                report.citations.append(chunk.citation)
        report.citations = _dedupe_citations(report.citations)

        # For fact-check and claim-audit questions, the evidence branch is
        # the primary source; NL2SQL provides only supplementary Reddit
        # corroboration.  Do not bail out on SQL failure when evidence
        # chunks were successfully retrieved — the LLM can still compose
        # a meaningful answer from the official-source evidence alone.
        _fact_check_with_evidence = (
            evidence_outs
            and (_question_is_fact_check(rq) or _question_is_topic_claim_audit(rq))
        )

        if (_planned_branch(execution, "nl2sql") and not sql_outs
                and _question_needs_sql(rq)):
            if not ((kg_outs and _question_needs_kg(rq)) or _fact_check_with_evidence):
                report.markdown_body = (
                    "I couldn't answer the requested topic/count/emotion "
                    "query because the NL2SQL branch failed before returning "
                    "rows. The structured output includes the branch error."
                )
                report.needs_human_review = True
                report.notes.append("nl2sql_required_but_failed")
                return report
            if _fact_check_with_evidence:
                report.notes.append("nl2sql_failed_but_evidence_available")
            else:
                report.notes.append("nl2sql_failed_but_kg_available")

        if _question_needs_sql(rq) and sql_outs:
            successful_sql = [s for s in sql_outs if s.success]
            if not successful_sql:
                if not ((kg_outs and _question_needs_kg(rq)) or _fact_check_with_evidence):
                    report.markdown_body = (
                        "The NL2SQL branch failed before returning rows. I "
                        "won't invent Reddit/community facts; check the "
                        "structured SQL output for the failed attempts."
                    )
                    report.needs_human_review = True
                    report.notes.append("nl2sql_failed")
                    return report
                if _fact_check_with_evidence:
                    report.notes.append("nl2sql_failed_but_evidence_available")
                else:
                    report.notes.append("nl2sql_failed_but_kg_available")
            elif not any(s.rows for s in successful_sql):
                if not ((kg_outs and _question_needs_kg(rq)) or _fact_check_with_evidence):
                    report.markdown_body = (
                        "I ran the SQL query for this request, but it "
                        "returned no matching rows. I won't invent topics or "
                        "counts; check the structured SQL output for the "
                        "exact filter that produced the empty result."
                    )
                    report.needs_human_review = True
                    report.notes.append("nl2sql_empty_result")
                    return report
                if _fact_check_with_evidence:
                    report.notes.append("nl2sql_empty_but_evidence_available")
                else:
                    report.notes.append("nl2sql_empty_but_kg_available")

        if not (evidence_outs or sql_outs or kg_outs):
            report.markdown_body = (
                "I couldn't gather enough data to answer that. "
                "Please refine the question or check the run targeted."
            )
            report.notes.append("no_branch_output")
            return report

        if kg_outs and not any(_kg_has_signal(k) for k in kg_outs):
            report.notes.append("kg_empty_result")
        if (_question_is_fact_check(rq) and sql_outs
                and not any(s.rows for s in sql_outs)):
            report.notes.append("fact_check_reddit_rows_empty")
        if (_question_is_fact_check(rq) and evidence_outs
                and not any(ev.bundle.chunks for ev in evidence_outs)):
            report.notes.append("fact_check_official_chunks_empty")
        if _question_is_topic_claim_audit(rq):
            report.notes.append("topic_claim_audit")

        # Build the LLM payload
        user_payload = self._build_payload(
            rq, execution, evidence_outs, sql_outs, kg_outs, failed,
        )
        try:
            data = self._call_llm(user_payload)
        except Exception as exc:
            log.error("report_writer.llm_error", error=str(exc)[:160])
            report.markdown_body = self._fallback_body(
                evidence_outs, sql_outs, kg_outs,
            )
            report.notes.append(f"llm_error: {exc}")
            return report

        body = (data.get("markdown_body") or "").strip()
        numbers_raw = data.get("numbers") or []

        if not body:
            report.markdown_body = self._fallback_body(
                evidence_outs, sql_outs, kg_outs,
            )
            report.notes.append("empty_markdown_body")
            return report

        # Post-process: swap raw topic/entity ids for human labels unless the
        # user explicitly requested those ids.
        body = self._humanize_ids(
            body,
            preserve_topic_ids=_question_requests_id(rq, "topic_id"),
            preserve_entity_ids=_question_requests_id(rq, "entity_id"),
        )
        body = _expand_inline_evidence_citations(body, evidence_outs)
        body = _apply_fact_check_data_guards(
            body, rq, evidence_outs, sql_outs,
        )
        body = _apply_topic_sentiment_guard(body, sql_outs)
        body = _apply_kg_data_guards(body, rq, kg_outs)

        report.markdown_body = body
        for n in numbers_raw:
            if not isinstance(n, dict):
                continue
            try:
                report.numbers.append(ReportNumber(
                    label=str(n.get("label") or "")[:80],
                    value=float(n.get("value") or 0.0),
                    source_branch=str(n.get("source_branch") or "")[:20],
                    source_ref=(str(n.get("source_ref"))[:160]
                                if n.get("source_ref") is not None else None),
                ))
            except (TypeError, ValueError):
                continue

        log.info("report_writer.done",
                 branches=report.branches_used,
                 chars=len(report.markdown_body),
                 numbers=len(report.numbers),
                 citations=len(report.citations))
        return report

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_payload(
        rq: RewrittenQuery,
        execution: PlanExecutionV2,
        evidence_outs: list[EvidenceOutput],
        sql_outs: list[SQLOutput],
        kg_outs: list[KGOutput],
        failed: list[Any],
    ) -> str:
        sections: list[str] = [f"User question: {rq.original}"]
        fact_check = _question_is_fact_check(rq)
        claim_audit = _question_is_topic_claim_audit(rq)
        comparison_or_recap = (
            not fact_check
            and not claim_audit
            and _question_is_comparison_or_recap(rq)
        )

        if fact_check:
            sections.append(
                "Fact-check mode: write a report-style answer with an explicit "
                "official-source verdict. Use only evidence chunks for "
                "official verification and only non-empty SQL rows for Reddit "
                "corroboration."
            )
        if claim_audit:
            sections.append(
                "Topic claim audit mode: extract claims only from Reddit SQL "
                "rows. For each claim, include author and post_id, classify it "
                "as consistent, contradicted, or insufficient evidence, and "
                "cite evidence chunk_ids for any official/evidence judgment."
            )
        if comparison_or_recap:
            sections.append(
                "Comparison/recap mode: structure the answer with two "
                "headings - '### Official Media Recap' (summarize the "
                "evidence chunks with inline source citations) and "
                "'### Community Perspective' (summarize the SQL rows). Do "
                "NOT apply the fact-check verdict labels. If one side has no "
                "data, keep its heading and explicitly state that nothing was "
                "retrieved for it."
            )

        if rq.subtasks:
            sections.append("Subtasks:")
            for i, s in enumerate(rq.subtasks, start=1):
                sections.append(
                    f"  {i}. ({s.intent}) {s.text} "
                    f"[branches: {','.join(s.suggested_branches) or 'auto'}]"
                )

        evidence_chunk_count = sum(
            len(ev.bundle.chunks) for ev in evidence_outs
        )
        if fact_check or claim_audit:
            sections.append(
                f"Official evidence availability: {evidence_chunk_count} "
                "retrieved chunk(s)."
            )

        if evidence_outs:
            sections.append("Evidence chunks:")
            evidence_cap = 12 if claim_audit else 8
            for ev in evidence_outs:
                for c in ev.bundle.chunks[:evidence_cap]:
                    sections.append(
                        f"  - evidence_id={c.chunk_id}; "
                        f"source={c.citation.source}; "
                        f"domain={c.citation.domain}; "
                        f"title={c.citation.title!r}; "
                        f"url={c.citation.url}; "
                        f"publish_date={c.citation.publish_date}; "
                        f"text={c.text[:500]}"
                    )

        if failed:
            sections.append("Branch failures:")
            for f in failed:
                sections.append(
                    f"  - {f.invocation.branch}: "
                    f"{f.status.error_kind or 'error'} "
                    f"{(f.status.error or '')[:200]}"
                )

        if sql_outs:
            sections.append("SQL outputs:")
            for s in sql_outs:
                if not s.success:
                    sections.append(
                        f"  - [SQL FAILED] attempts={len(s.attempts)} "
                        f"error_kind={s.attempts[-1].error_kind if s.attempts else '-'}"
                    )
                    continue
                sections.append(
                    f"  - SQL: {s.final_sql}"
                )
                if s.rows:
                    row_cap = 20
                    sample = s.rows[:row_cap]
                    sections.append(
                        f"    rows ({len(s.rows)} total, showing up to "
                        f"{row_cap}): "
                        f"{json.dumps(sample, default=str)[:12000]}"
                    )
                else:
                    sections.append(
                        "    rows (0 total): []"
                    )
                    if fact_check or claim_audit:
                        sections.append(
                            "    Reddit/community availability: no matching "
                            "rows were retrieved for this SQL output."
                        )

        if kg_outs:
            sections.append("Graph outputs:")
            for k in kg_outs:
                has_returned_entities = bool(k.nodes or k.edges)
                sections.append(
                    f"  - kind={k.query_kind} target={k.target} "
                    f"metrics={k.metrics} "
                    f"returned_entities="
                    f"{'yes' if has_returned_entities else 'none'}"
                )
                if not has_returned_entities and k.metrics:
                    sections.append(
                        "    NOTE: any 'analyzed_node_count' / "
                        "'analyzed_edge_count' values describe the size of "
                        "the input graph the algorithm ran over, NOT entities "
                        "returned as query results. Do NOT phrase those "
                        "numbers as 'nodes/participants found' or 'edges "
                        "returned'."
                    )
                if k.target.get("fallback_from_topic_id"):
                    sections.append(
                        "    topic_fallback: primary topic "
                        f"{k.target.get('fallback_from_topic_id')} had no "
                        "KG graph signal; this KG result uses semantically "
                        f"similar topic {k.target.get('selected_topic_id')}."
                    )
                if not _kg_has_signal(k):
                    sections.append(
                        "    graph_availability: no usable graph nodes, "
                        "edges, or positive metrics. Treat this as "
                        "insufficient graph data, not evidence that the "
                        "network pattern is absent."
                    )
                if k.nodes:
                    sample = [
                        {"id": n.id, "label": n.label,
                         "props": dict(list(n.properties.items())[:5])}
                        for n in k.nodes[:8]
                    ]
                    sections.append(
                        f"    nodes: {json.dumps(sample, default=str)[:600]}"
                    )
                    # Surface Entity nodes explicitly so the writer names them
                    entity_nodes = [n for n in k.nodes if n.label == "Entity"]
                    if entity_nodes:
                        entity_names = [
                            str(n.properties.get("name") or n.id)
                            for n in entity_nodes[:15]
                        ]
                        sections.append(
                            f"    key_entities_in_graph: "
                            f"{', '.join(entity_names)}"
                        )
                if k.edges:
                    if k.query_kind == "reply_chains":
                        node_lookup = {n.id: n for n in k.nodes}
                        edge_sample = []
                        for e in k.edges[:12]:
                            src = node_lookup.get(e.source_id)
                            dst = node_lookup.get(e.target_id)
                            if src and dst:
                                edge_sample.append(
                                    f"{_node_excerpt(src)} replied to "
                                    f"{_node_excerpt(dst)}"
                                )
                            else:
                                edge_sample.append(
                                    f"{e.source_id} --{e.rel_type}--> "
                                    f"{e.target_id}"
                                )
                    else:
                        edge_sample = [
                            f"{e.source_id} --{e.rel_type}--> {e.target_id}"
                            for e in k.edges[:8]
                        ]
                    sections.append(
                        "    edges: " + "; ".join(edge_sample)[:2400]
                    )

        return "\n".join(sections)

    def _call_llm(self, user_payload: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=3000,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _WRITER_SYSTEM},
                {"role": "user", "content": user_payload[:16000]},
            ],
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        return json.loads(raw)

    @staticmethod
    def _humanize_ids(
        body: str,
        *,
        preserve_topic_ids: bool = False,
        preserve_entity_ids: bool = False,
    ) -> str:
        """Replace raw `topic_*` / `ent_*` ids with their PG label / name.

        Best-effort: if Postgres is unreachable, return the original body
        untouched. Misses are left as-is (no exceptions raised).
        """
        topic_ids = set(_TOPIC_ID_RE.findall(body))
        entity_ids = set(_ENTITY_ID_RE.findall(body))
        if not (topic_ids or entity_ids):
            return body
        try:
            from services.postgres_service import PostgresService
            pg = PostgresService()
            pg.connect()
        except Exception:
            return body

        replacements: dict[str, str] = {}
        try:
            with pg.cursor() as cur:
                if topic_ids:
                    cur.execute(
                        "SELECT topic_id, label FROM topics_v2 "
                        "WHERE topic_id = ANY(%s)",
                        (list(topic_ids),),
                    )
                    for row in cur.fetchall():
                        label = (row.get("label") or "").strip()
                        if label:
                            if not preserve_topic_ids:
                                replacements[row["topic_id"]] = label
                if entity_ids:
                    cur.execute(
                        "SELECT entity_id, name FROM entities_v2 "
                        "WHERE entity_id = ANY(%s)",
                        (list(entity_ids),),
                    )
                    for row in cur.fetchall():
                        name = (row.get("name") or "").strip()
                        if name:
                            if not preserve_entity_ids:
                                replacements[row["entity_id"]] = name
        except Exception as exc:
            log.warning("report_writer.humanize_ids_error",
                        error=str(exc)[:120])
            return body

        if not replacements:
            return body

        def _sub(match: "re.Match") -> str:
            rid = match.group(0)
            return replacements.get(rid, rid)

        if not preserve_topic_ids:
            body = _TOPIC_ID_RE.sub(_sub, body)
        if not preserve_entity_ids:
            body = _ENTITY_ID_RE.sub(_sub, body)
        # Strip cosmetic prefixes like "Topic ID: <label>" -> "<label>"
        if not preserve_topic_ids:
            body = re.sub(r"\bTopic ID:\s*", "", body)
        if not preserve_entity_ids:
            body = re.sub(r"\bEntity ID:\s*", "", body)
        return body

    @staticmethod
    def _fallback_body(
        evidence_outs: list[EvidenceOutput],
        sql_outs: list[SQLOutput],
        kg_outs: list[KGOutput],
    ) -> str:
        bits: list[str] = ["I gathered the following data but couldn't compose"
                            " a polished answer:"]
        if evidence_outs:
            n = sum(len(e.bundle.chunks) for e in evidence_outs)
            bits.append(f"- {n} evidence chunks from official sources")
        if sql_outs:
            ok = sum(1 for s in sql_outs if s.success)
            bits.append(f"- {ok} successful SQL queries on community data")
        if kg_outs:
            bits.append(f"- {len(kg_outs)} knowledge-graph results")
        return "\n".join(bits)


def _filter_outputs(execution: PlanExecutionV2, branch: str, model_cls):
    out: list[Any] = []
    for r in execution.results:
        if r.invocation.branch != branch:
            continue
        if not r.status.success or r.output is None:
            continue
        try:
            out.append(model_cls.model_validate(r.output))
        except Exception:
            continue
    return out


def _failed_results(execution: PlanExecutionV2):
    return [r for r in execution.results if not r.status.success]


def _planned_branch(execution: PlanExecutionV2, branch: str) -> bool:
    return any(inv.branch == branch for inv in execution.workflow)


def _question_needs_sql(rq: RewrittenQuery) -> bool:
    sql_intents = {
        "community_count", "community_listing", "trend",
        "topic_claim_audit",
    }
    if any(s.intent in sql_intents for s in rq.subtasks):
        return True
    text = (rq.original or "").lower()
    return any(k in text for k in (
        "topic", "topics", "count", "counts", "emotion", "sentiment",
        "post count", "dominant emotion",
    ))


def _question_needs_kg(rq: RewrittenQuery) -> bool:
    kg_intents = {
        "propagation", "propagation_trace", "influencer_query",
        "coordination_check", "community_structure", "cascade_query",
        "explain_decision",
    }
    if any(s.intent in kg_intents for s in rq.subtasks):
        return True
    text = " ".join([rq.original] + [s.text for s in rq.subtasks]).lower()
    return any(k in text for k in (
        "knowledge graph", "kg", "propagation", "reply chain",
        "reply chains", "path", "paths", "amplifying", "influencer",
        "network", "centrality", "echo chamber", "community structure",
    ))


def _question_is_fact_check(rq: RewrittenQuery) -> bool:
    if any(s.intent == "topic_claim_audit" for s in rq.subtasks):
        return False
    # Comparison / official-recap intents share keywords with fact-check
    # ("official source", "official media", ...). When the Rewriter has
    # already classified the question as one of those, trust that signal
    # and stay out of fact-check mode - otherwise both branches add their
    # own system addenda and the LLM gets contradictory instructions.
    if any(s.intent in {"official_recap", "comparison"} for s in rq.subtasks):
        return False
    if any(s.intent == "fact_check" for s in rq.subtasks):
        return True
    text = " ".join([rq.original] + [s.text for s in rq.subtasks]).lower()
    return any(k in text for k in (
        "fact-check", "fact check", "verify", "verification",
        "official source", "official sources", "claim", "true or false",
        "真假", "真伪", "核实", "事实核查", "官方来源", "官方说法",
    ))


def _question_is_comparison_or_recap(rq: RewrittenQuery) -> bool:
    """Detect comparison / official-recap questions.

    Trust the Rewriter's intent first. The text-keyword fallback is
    deliberately narrow - it must NOT overlap with fact-check keywords
    like "official source(s)" or "verify", or both _question_is_fact_check
    and this helper would fire and the LLM would receive contradictory
    addenda.
    """
    if any(s.intent in {"official_recap", "comparison"} for s in rq.subtasks):
        return True
    text = " ".join([rq.original] + [s.text for s in rq.subtasks]).lower()
    return any(k in text for k in (
        "official media", "what does official media", "what do official media",
        "what are official media",
        " vs ", "versus", "compare ", "comparison between",
        "contrast",
        "官方媒体",
    ))


def _question_is_topic_claim_audit(rq: RewrittenQuery) -> bool:
    if any(s.intent == "topic_claim_audit" for s in rq.subtasks):
        return True
    text = " ".join([rq.original] + [s.text for s in rq.subtasks]).lower()
    has_claims = (
        "claim" in text or "claims" in text
        or "说法" in text or "观点" in text
    )
    has_topic = "topic" in text or "主题" in text
    has_official = (
        "official" in text or "evidence" in text
        or "官方" in text or "证据" in text
    )
    has_buckets = any(k in text for k in (
        "consistent", "contradict", "insufficient",
        "agree", "conflict", "一致", "矛盾", "证据不足", "无法判断",
    ))
    return has_claims and has_topic and has_official and has_buckets


def _question_requests_id(rq: RewrittenQuery, field_name: str) -> bool:
    text = " ".join([rq.original] + [s.text for s in rq.subtasks]).lower()
    return field_name.lower() in text


def _kg_has_signal(k: KGOutput) -> bool:
    if k.nodes or k.edges:
        return True
    for key, value in k.metrics.items():
        if key in {"cache_hit", "since_days", "reason"}:
            continue
        if isinstance(value, (int, float)) and value > 0:
            return True
        if value not in (0, 0.0, None, "", [], {}):
            return True
    return False


def _node_excerpt(node: Any) -> str:
    props = getattr(node, "properties", {}) or {}
    author = props.get("author") or props.get("author_id") or node.id
    text = (props.get("text") or "").strip()
    if len(text) > 90:
        text = text[:87] + "..."
    if text:
        return f"{author} (\"{text}\")"
    return str(author)


def _apply_fact_check_data_guards(
    body: str,
    rq: RewrittenQuery,
    evidence_outs: list[EvidenceOutput],
    sql_outs: list[SQLOutput],
) -> str:
    if not _question_is_fact_check(rq):
        return body

    additions: list[str] = []
    if evidence_outs and not any(ev.bundle.chunks for ev in evidence_outs):
        additions.append(
            "Official-source check: no official-source evidence chunks were "
            "retrieved, so the claim cannot be verified from official sources "
            "in this run."
        )
    if sql_outs and not any(s.rows for s in sql_outs):
        additions.append(
            "Reddit/community check: SQL returned 0 matching rows, so no "
            "matching Reddit posts or comments were retrieved in this run."
        )

    if not additions:
        return body

    lower_body = body.lower()
    missing = [
        line for line in additions
        if line.lower()[:48] not in lower_body
    ]
    if not missing:
        return body
    return body.rstrip() + "\n\n" + "\n".join(missing)


def _apply_topic_sentiment_guard(
    body: str,
    sql_outs: list[SQLOutput],
) -> str:
    """Correct a common report error: ignoring topic-level emotion fallback."""
    emotion = _best_sql_emotion(sql_outs)
    if not emotion:
        return body
    lower = body.lower()
    if "dominant emotion" not in lower or "not specified" not in lower:
        return body

    replacement = (
        f"**Dominant Emotion:** {emotion} "
        "(topic-level aggregate; post-level rows may be unannotated)."
    )
    updated = re.sub(
        r"(?im)^(\s*[-*]\s*)?\*\*Dominant Emotion\*\*:.*$",
        lambda m: f"{m.group(1) or ''}{replacement}",
        body,
        count=1,
    )
    if updated != body:
        return updated
    updated = re.sub(
        r"(?is)dominant emotion[^.\n]*not specified[^.\n]*(?:\.|$)",
        f"dominant emotion is {emotion} at the topic level. ",
        body,
        count=1,
    )
    return updated


def _best_sql_emotion(sql_outs: list[SQLOutput]) -> str:
    for sql_out in sql_outs:
        for row in sql_out.rows:
            for key in (
                "dominant_emotion",
                "topic_dominant_emotion",
                "topic_emotion",
            ):
                value = str(row.get(key) or "").strip()
                if value and value.lower() not in {
                    "not specified", "null", "none", "unknown",
                }:
                    return value
    return ""


def _apply_kg_data_guards(
    body: str,
    rq: RewrittenQuery,
    kg_outs: list[KGOutput],
) -> str:
    """Prevent missing KG data from being phrased as a negative finding."""
    if not kg_outs or any(_kg_has_signal(k) for k in kg_outs):
        return body
    text = " ".join([rq.original] + [s.text for s in rq.subtasks]).lower()
    kg_sensitive = any(k in text for k in (
        "echo chamber", "cluster", "clusters", "community",
        "propagation", "reply chain", "network",
    ))
    if not kg_sensitive:
        return body

    limitation = (
        "The KG branch returned no usable nodes, edges, or positive graph "
        "metrics for this topic, so this run cannot determine whether the "
        "discussion forms an echo chamber or distinct graph clusters."
    )
    replacements = [
        r"(?is)the discussion does not form an echo chamber[^.\n]*(?:\.|$)",
        r"(?is)the discussion does not exhibit characteristics of an echo chamber[^.\n]*(?:\.|$)",
        r"(?is)the topic does not form an echo chamber[^.\n]*(?:\.|$)",
    ]
    updated = body
    for pattern in replacements:
        updated = re.sub(pattern, limitation, updated)
    if limitation.lower()[:80] in updated.lower():
        return updated
    return updated.rstrip() + "\n\n" + limitation


def _expand_inline_evidence_citations(
    body: str,
    evidence_outs: list[EvidenceOutput],
) -> str:
    chunks: dict[str, Any] = {}
    for ev in evidence_outs:
        for c in ev.bundle.chunks:
            chunks[c.chunk_id] = c
    if not chunks:
        return body

    def _replace(match: "re.Match") -> str:
        chunk_id = match.group(1)
        chunk = chunks.get(chunk_id)
        if chunk is None:
            return match.group(0)
        citation = chunk.citation
        title = citation.title or chunk.text[:80]
        source = citation.source or citation.domain or "evidence"
        date = ""
        if citation.publish_date:
            try:
                date = citation.publish_date.date().isoformat()
            except Exception:
                date = str(citation.publish_date)[:10]
        parts = [source, f"\"{title}\""]
        if date:
            parts.append(date)
        if citation.url:
            parts.append(citation.url)
        return "(Evidence: " + ", ".join(parts) + ")"

    return re.sub(
        r"\[(?:chunk_id|evidence_id|source_ref)?\s*[:=]?\s*"
        r"((?:[0-9a-f]{12,}|c[\w:-]*|ev[\w:-]*|chunk_[\w:-]+))\]",
        _replace,
        body,
        flags=re.IGNORECASE,
    )


def _dedupe_citations(items: list[Citation]) -> list[Citation]:
    seen: set[str] = set()
    out: list[Citation] = []
    for c in items:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out
