"""Chat API request/response models.

These are the wire-format objects exchanged between the UI and
`POST /chat/query`. They intentionally carry the *structured*
capability output in addition to the human-readable `answer_text`, so
the UI can render rich widgets (evidence cards, metric tables, visuals)
without re-deriving them from the text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    """One chat message as the UI displays it (no structured payload)."""
    role: str                            # "user" | "assistant"
    content: str
    at: datetime = Field(default_factory=_utcnow)


class ChatQuery(BaseModel):
    """Inbound request body for POST /chat/query."""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Outbound response body for POST /chat/query.

    `capability_output` is the raw Pydantic output from the capability,
    serialized to dict. The UI uses it for rich rendering; callers that
    only want the text can read `answer_text`.

    redesign-2026-05 Phase 4: v2 path also fills `branches_used`,
    `branch_outputs`, `citations`, and `needs_human_review`. v1 fields are
    kept populated where possible for backwards compatibility.
    """
    session_id: str
    answer_text: str
    capability_used: Optional[str] = None
    capability_output: Optional[dict[str, Any]] = None
    visual_paths: list[str] = Field(default_factory=list)
    # v2 additions
    branches_used: list[str] = Field(default_factory=list)
    branch_outputs: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    needs_human_review: bool = False
    critic_attempts: int = Field(default=1)
    critic_passed_on: Optional[str] = Field(default=None)  # "first" | "second" | None
