"""
Claim extraction agent — first-class atomic claims for fact-check / topic-claim-audit.

Design:
- Reddit submissions (post_id NOT starting with 'reddit_c_') carry the canonical
  claim text in their title. We run an LLM extractor on submission text to
  produce 0-N atomic claims.
- Reddit comments (post_id 'reddit_c_*') don't usually assert new factual
  claims; they react to the parent submission's claim. We link comments to
  their parent's claims via post_claims_v2 with role='discusses'.
- Within the batch, near-duplicate claims (simhash hamming <= 3) are merged
  so paraphrased claims share one claims_v2 row.

Output:
- list[ClaimRecord]            — unique claims with embedding + simhash
- list[PostClaimLink]           — (post_id, claim_id, role) edges
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import openai
import structlog

from agents.post_dedup import compute_simhash, hamming_distance
from config import OPENAI_API_KEY, OPENAI_MODEL
from models.post import Post
from services.embeddings_service import EmbeddingsService

log = structlog.get_logger(__name__)


_EXTRACT_SYSTEM = """You extract atomic factual claims from Reddit submission titles.

A "claim" is a factual proposition that could in principle be verified against
authoritative sources (news outlets, official statements, etc.).

Rules:
- Drop the [domain.com] prefix if present.
- One title may yield 0, 1, or rarely 2 claims.
- Skip pure opinion ("X is amazing"), pure questions ("Did X happen?"), and
  pure editorial framing ("Why X is wrong").
- Strip first-person framing — emit the underlying factual proposition.
- Keep each claim under 25 words.
- Do NOT invent details that are not in the title.

Output STRICT JSON:
{"claims": ["<claim 1>", "<claim 2>", ...]}

If the title contains no verifiable factual claim, output {"claims": []}.
"""


_DOMAIN_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")


def _strip_domain_prefix(text: str) -> str:
    return _DOMAIN_PREFIX_RE.sub("", text or "").strip()


def _normalize_for_id(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _claim_id(text: str) -> str:
    digest = hashlib.sha256(_normalize_for_id(text).encode("utf-8")).hexdigest()
    return f"claim_{digest[:12]}"


@dataclass
class ClaimRecord:
    claim_id: str
    claim_text: str
    topic_id: Optional[str] = None
    embedding: Optional[list[float]] = None
    simhash: Optional[int] = None
    source_url: Optional[str] = None
    role: str = "asserts"            # default role on the source post
    confidence: float = 0.7


@dataclass
class PostClaimLink:
    post_id: str
    claim_id: str
    role: str = "asserts"            # asserts | discusses | rebuts | parent_context


@dataclass
class ClaimExtractionResult:
    claims: list[ClaimRecord] = field(default_factory=list)
    links: list[PostClaimLink] = field(default_factory=list)


class ClaimExtractor:
    def __init__(
        self,
        embeddings: Optional[EmbeddingsService] = None,
        *,
        client: Optional[openai.OpenAI] = None,
        model: str = OPENAI_MODEL,
        simhash_merge_threshold: int = 3,
    ) -> None:
        self._embeddings = embeddings or EmbeddingsService()
        self._client = client or openai.OpenAI(api_key=OPENAI_API_KEY)
        self._model = model
        self._merge_threshold = simhash_merge_threshold

    def extract(self, posts: list[Post]) -> ClaimExtractionResult:
        result = ClaimExtractionResult()
        if not posts:
            return result

        post_by_id: dict[str, Post] = {p.id: p for p in posts}
        submissions = [p for p in posts if not p.id.startswith("reddit_c_")]
        comments = [p for p in posts if p.id.startswith("reddit_c_")]

        # ── Stage 1: extract claims from each submission ─────────────────────
        # claim_id -> ClaimRecord
        unique_claims: dict[str, ClaimRecord] = {}
        # post_id -> set of claim_ids (de-dupes if same claim appears from
        # multiple LLM passes on the same post)
        post_to_claims: dict[str, set[str]] = {}

        for sub in submissions:
            text = _strip_domain_prefix(sub.text or "")
            if len(text) < 10:
                continue
            claim_texts = self._llm_extract(text)
            if not claim_texts:
                continue
            url = self._extract_link(sub.text or "")
            for ct in claim_texts:
                cid = self._merge_or_register(
                    ct, unique_claims,
                    topic_id=sub.topic_id,
                    source_url=url,
                )
                post_to_claims.setdefault(sub.id, set()).add(cid)

        # ── Stage 2: link comments to their parent submission's claims ───────
        for c in comments:
            parent_id = getattr(c, "parent_post_id", None)
            if not parent_id:
                continue
            parent_claims = post_to_claims.get(parent_id)
            if not parent_claims:
                continue
            for cid in parent_claims:
                post_to_claims.setdefault(c.id, set()).add(cid)

        # ── Stage 3: batch-embed all unique claims ───────────────────────────
        if unique_claims:
            ordered = list(unique_claims.values())
            try:
                vectors = self._embeddings.embed_batch(
                    [c.claim_text for c in ordered]
                )
                for c, v in zip(ordered, vectors):
                    c.embedding = v
            except Exception as exc:
                log.error("claim_extractor.embed_error",
                          count=len(ordered), error=str(exc)[:160])

        # ── Stage 4: assemble output ─────────────────────────────────────────
        result.claims = list(unique_claims.values())
        for post_id, cids in post_to_claims.items():
            post = post_by_id.get(post_id)
            is_submission = post is not None and not post.id.startswith("reddit_c_")
            role = "asserts" if is_submission else "discusses"
            for cid in cids:
                result.links.append(PostClaimLink(
                    post_id=post_id, claim_id=cid, role=role,
                ))

        log.info("claim_extractor.done",
                 submissions=len(submissions),
                 comments=len(comments),
                 unique_claims=len(result.claims),
                 links=len(result.links))
        return result

    # ── LLM extraction ─────────────────────────────────────────────────────────

    def _llm_extract(self, text: str) -> list[str]:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            claims = data.get("claims") or []
            return [c.strip() for c in claims
                    if isinstance(c, str) and len(c.strip()) >= 8]
        except Exception as exc:
            log.warning("claim_extractor.llm_error",
                        text=text[:80], error=str(exc)[:160])
            return []

    # ── Dedup / merge by simhash ───────────────────────────────────────────────

    def _merge_or_register(
        self,
        claim_text: str,
        unique_claims: dict[str, "ClaimRecord"],
        *,
        topic_id: Optional[str],
        source_url: Optional[str],
    ) -> str:
        new_simhash = compute_simhash(claim_text)
        # Look for a near-duplicate already in the batch.
        for cid, existing in unique_claims.items():
            if existing.simhash is None:
                continue
            if hamming_distance(existing.simhash, new_simhash) <= self._merge_threshold:
                # Reuse existing record; keep first text we saw.
                return cid
        cid = _claim_id(claim_text)
        if cid not in unique_claims:
            unique_claims[cid] = ClaimRecord(
                claim_id=cid,
                claim_text=claim_text,
                topic_id=topic_id,
                simhash=new_simhash,
                source_url=source_url,
            )
        return cid

    # ── URL helper ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_link(raw: str) -> Optional[str]:
        m = re.search(r"https?://[^\s)]+", raw or "")
        return m.group(0) if m else None
