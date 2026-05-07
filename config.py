"""
Central configuration loaded from environment / .env.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))
KUZU_DB_DIR: str = os.getenv("KUZU_DB_DIR", str(DATA_DIR / "kuzu_graph"))
RUNS_DIR: str = os.getenv("RUNS_DIR", str(DATA_DIR / "runs"))

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set.\n"
        "Add OPENAI_API_KEY=sk-... to your .env file."
    )

POSTGRES_DSN: str = os.getenv(
    "POSTGRES_DSN", "postgresql://society:society_pass@localhost:5432/society_db"
)
POSTGRES_READONLY_DSN: str = os.getenv("POSTGRES_READONLY_DSN", "")

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIM: int = 1536

CHROMA_OFFICIAL_COLLECTION: str = os.getenv(
    "CHROMA_OFFICIAL_COLLECTION", "chroma_official"
)
CHROMA_NL2SQL_COLLECTION: str = os.getenv(
    "CHROMA_NL2SQL_COLLECTION", "chroma_nl2sql"
)
CHROMA_PLANNER_COLLECTION: str = os.getenv(
    "CHROMA_PLANNER_COLLECTION", "chroma_planner"
)

NL2SQL_CONFLICT_SIM_LOW: float = float(os.getenv("NL2SQL_CONFLICT_SIM_LOW", "0.92"))
NL2SQL_CONFLICT_SIM_HIGH: float = float(os.getenv("NL2SQL_CONFLICT_SIM_HIGH", "0.95"))
NL2SQL_MAX_REPAIR_ROUNDS: int = int(os.getenv("NL2SQL_MAX_REPAIR_ROUNDS", "3"))
NL2SQL_RESULT_ROW_LIMIT: int = int(os.getenv("NL2SQL_RESULT_ROW_LIMIT", "1000"))
NL2SQL_STATEMENT_TIMEOUT_MS: int = int(os.getenv("NL2SQL_STATEMENT_TIMEOUT_MS", "5000"))

SESSION_MAX_TURNS: int = int(os.getenv("SESSION_MAX_TURNS", "40"))
SESSION_MIN_TURNS_TO_COMPACT: int = int(
    os.getenv("SESSION_MIN_TURNS_TO_COMPACT", "10")
)
SESSION_SUMMARY_MAX_CHARS: int = int(
    os.getenv("SESSION_SUMMARY_MAX_CHARS", "1200")
)

EXPERIENCE_MIN_CONFIDENCE: float = float(
    os.getenv("EXPERIENCE_MIN_CONFIDENCE", "0.2")
)
EXPERIENCE_TTL_DAYS: int = int(os.getenv("EXPERIENCE_TTL_DAYS", "30"))

# KG topic workflows first anchor on the best-matching topic, then may fall
# back to semantically similar topics when the exact topic has no Kuzu subgraph.
KG_TOPIC_SEMANTIC_FALLBACK_MIN_SIM: float = float(
    os.getenv("KG_TOPIC_SEMANTIC_FALLBACK_MIN_SIM", "0.30")
)

REDDIT_PROXY: str = os.getenv("REDDIT_PROXY", "")
REDDIT_DEFAULT_SUBREDDITS: list[str] = [
    s.strip()
    for s in os.getenv(
        "REDDIT_DEFAULT_SUBREDDITS",
        "conspiracy,worldnews,politics,health,news",
    ).split(",")
    if s.strip()
]

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Maximum seconds to wait for a single OpenAI API call.  Must be well under
# RESEARCH_API_LONG_TIMEOUT (240 s) so the UI never sees a read-timeout.
LLM_REQUEST_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "180"))

MULTIMODAL_DAILY_BUDGET_USD: float = float(
    os.getenv("MULTIMODAL_DAILY_BUDGET_USD", "5.0")
)
MULTIMODAL_COST_PER_CALL_USD: float = float(
    os.getenv("MULTIMODAL_COST_PER_CALL_USD", "0.02")
)
MULTIMODAL_MIN_LIKES: int = int(os.getenv("MULTIMODAL_MIN_LIKES", "50"))
MULTIMODAL_MIN_REPLIES: int = int(os.getenv("MULTIMODAL_MIN_REPLIES", "20"))
