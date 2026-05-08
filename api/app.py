"""FastAPI entrypoint — exposes read-only endpoints over data/runs/*.

Run with:
    uvicorn api.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import admin_import, admin_nl2sql, artifacts, chat, health, plan, reflection, retrieve, runs


def _resolve_runs_root() -> Path:
    env = os.getenv("RUNS_DIR")
    if env:
        return Path(env)
    # Fall back to <repo>/data/runs, which is where the pipeline writes
    return Path(__file__).resolve().parent.parent / "data" / "runs"


RUNS_ROOT = _resolve_runs_root()
# Ensure RUNS_ROOT exists at startup so /health returns ok=true on a
# fresh clone before any pipeline run has written its first manifest.
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Society Analysis — Research API",
    description="Read-only access to pipeline run artifacts.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    # Chat endpoint needs POST; read-only routes still only use GET.
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.state.runs_root = RUNS_ROOT

app.include_router(runs.router)
app.include_router(artifacts.router)
app.include_router(chat.router)
app.include_router(retrieve.router)
app.include_router(reflection.router)
app.include_router(health.router)
app.include_router(plan.router)
app.include_router(admin_import.router)
app.include_router(admin_nl2sql.router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "society-analysis research-api",
        "runs_root": str(RUNS_ROOT),
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": RUNS_ROOT.exists(), "runs_root": str(RUNS_ROOT)}
