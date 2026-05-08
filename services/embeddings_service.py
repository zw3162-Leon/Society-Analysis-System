"""
Embeddings service — wraps OpenAI text-embedding-3-small.
Used for cross-modal text-image retrieval and claim deduplication.
"""
from __future__ import annotations

import structlog
from openai import OpenAI

from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM

log = structlog.get_logger(__name__)


class EmbeddingsService:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._model = EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a 1536-dim vector."""
        text = text.replace("\n", " ").strip()
        if not text:
            return [0.0] * EMBEDDING_DIM
        try:
            response = self._client.embeddings.create(
                input=[text],
                model=self._model,
            )
            return response.data[0].embedding
        except Exception as exc:
            log.error("embeddings.error", text=text[:80], error=str(exc))
            return [0.0] * EMBEDDING_DIM

    # OpenAI embeddings API caps a single request at 2048 inputs.
    _MAX_INPUTS_PER_REQUEST = 2000

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, chunking under the API's 2048-input cap."""
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        safe = [t if t else "." for t in cleaned]
        vecs: list[list[float]] = [[0.0] * EMBEDDING_DIM for _ in texts]
        chunk = self._MAX_INPUTS_PER_REQUEST
        for start in range(0, len(safe), chunk):
            end = min(start + chunk, len(safe))
            try:
                response = self._client.embeddings.create(
                    input=safe[start:end],
                    model=self._model,
                )
                for offset, item in enumerate(response.data):
                    vecs[start + offset] = item.embedding
            except Exception as exc:
                log.error(
                    "embeddings.batch_error",
                    count=end - start,
                    start=start,
                    error=str(exc),
                )
                # leave the chunk's slots as zero vectors (already initialized)
        for i, t in enumerate(cleaned):
            if not t:
                vecs[i] = [0.0] * EMBEDDING_DIM
        return vecs
