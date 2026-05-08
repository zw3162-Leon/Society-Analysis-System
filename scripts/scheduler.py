"""
scripts/scheduler.py - redesign-2026-05 v2 scheduler.

Drives the v2 backend pipelines on a cron schedule. Reads
`config/scheduler_tasks.yaml` for task definitions. Two execution modes:

  apscheduler  (default)  Long-running BlockingScheduler. Each task fires
                          on its cron expression. If `run_on_start: true`,
                          every enabled task also runs ONCE at startup so
                          the first deploy populates the stores with no
                          manual step.

  once                    Run a single task immediately and exit. Use this
                          when the host OS already has cron / Task
                          Scheduler / systemd timer driving the cadence.

Supported task kinds (kind field in YAML):
  - official_ingestion  -> agents.official_ingestion_pipeline.run_once
  - precompute_v2       -> agents.precompute_pipeline_v2.PrecomputePipelineV2.run
  - decay_experience    -> scripts.decay_chroma_experience.sweep_collection
  - seed_planner        -> scripts.seed_planner_memory.main

Usage:
    # Long-running daemon (runs once at startup, then on cron):
    python -m scripts.scheduler

    # Run every enabled task once and exit (post-deploy bootstrap):
    python -m scripts.scheduler --bootstrap

    # Run a single task once and exit:
    python -m scripts.scheduler --once --task official_sources

    # Show what would run (no execution):
    python -m scripts.scheduler --list
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

import config

log = structlog.get_logger(__name__)


# ── Task definitions ─────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    name: str
    kind: str
    cron: Optional[str] = None
    args: dict = field(default_factory=dict)
    enabled: bool = True
    run_on_start: bool = True


def load_tasks(yaml_path: Optional[Path] = None) -> list[TaskSpec]:
    yaml_path = yaml_path or (
        config.BASE_DIR / "config" / "scheduler_tasks.yaml"
    )
    if not yaml_path.exists():
        log.warning("scheduler.config_missing", path=str(yaml_path))
        return []
    try:
        import yaml  # type: ignore
    except ImportError:
        log.error("scheduler.pyyaml_missing",
                  hint="pip install pyyaml")
        return []

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults") or {}
    tasks: list[TaskSpec] = []
    for entry in (raw.get("tasks") or []):
        if not isinstance(entry, dict):
            continue
        merged = {**defaults, **entry}
        tasks.append(TaskSpec(
            name=merged["name"],
            kind=merged["kind"],
            cron=merged.get("cron"),
            args=dict(merged.get("args") or {}),
            enabled=bool(merged.get("enabled", True)),
            run_on_start=bool(merged.get("run_on_start", True)),
        ))
    return tasks


# ── Task runners ─────────────────────────────────────────────────────────────

def _run_official_ingestion(args: dict) -> dict:
    from agents.official_ingestion_pipeline import OfficialIngestionPipeline
    pipeline = OfficialIngestionPipeline(
        write_chroma=bool(args.get("write_chroma", True)),
    )
    return pipeline.run_once(source_filter=args.get("source_filter"))


def _run_precompute_v2(args: dict) -> dict:
    from agents.claim_extractor import ClaimExtractor
    from agents.entity_extractor import EntityExtractor
    from agents.ingestion import IngestionAgent
    from agents.knowledge import KnowledgeAgent
    from agents.multimodal_agent import MultimodalAgent
    from agents.post_dedup import PostDeduper
    from agents.precompute_pipeline_v2 import PrecomputePipelineV2
    from agents.schema_agent import SchemaAgent
    from agents.topic_clusterer import TopicClusterer
    from services.claude_vision_service import ClaudeVisionService
    from services.kuzu_service import KuzuService
    from services.manifest_service import ManifestService
    from services.postgres_service import PostgresService
    from services.reddit_service import RedditService
    from services.schema_sync import SchemaSync

    pg = PostgresService()
    # Writer instance: scheduled pipeline run mutates Kuzu.
    kuzu = KuzuService(read_only=False)
    vision = ClaudeVisionService()
    reddit = None
    try:
        reddit = RedditService()
    except Exception as exc:
        log.warning("scheduler.reddit_unavailable", error=str(exc)[:120])

    ingestion = IngestionAgent(
        pg=pg, kuzu=kuzu, vision=vision, reddit=reddit,
    )

    pipeline = PrecomputePipelineV2(
        ingestion=ingestion,
        knowledge=KnowledgeAgent(),
        multimodal=MultimodalAgent(),
        entity_extractor=EntityExtractor(),
        topic_clusterer=TopicClusterer(),
        post_deduper=PostDeduper(),
        schema_agent=SchemaAgent(),
        schema_sync=SchemaSync(pg=pg),
        claim_extractor=ClaimExtractor(),
        pg=pg,
        kuzu=kuzu,
    )

    ms = ManifestService()
    subreddits = args.get("subreddits") or None
    manifest = ms.new_run(
        query_text=(args.get("reddit_query")
                     or (",".join(subreddits) if subreddits else "")),
        subreddits=subreddits,
        reddit_query=args.get("reddit_query"),
        reddit_sort="hot",
        jsonl_path=args.get("jsonl_path"),
        image_url=None,
        image_path=None,
        days_back=int(args.get("reddit_days_back", 1)),
    )
    run_dir = ms.run_dir(manifest.run_id)
    result = pipeline.run(
        run_dir=run_dir,
        subreddits=subreddits,
        reddit_query=args.get("reddit_query"),
        reddit_days_back=int(args.get("reddit_days_back", 1)),
        jsonl_path=args.get("jsonl_path"),
    )
    return {
        "run_id": result.run_id,
        "posts": len(result.posts),
        "topics": len(result.topics),
        "stages": [(s.name, s.status) for s in result.stages],
    }


def _run_decay_experience(args: dict) -> dict:
    from scripts.decay_chroma_experience import sweep_collection
    from services.chroma_collections import ChromaCollections
    cols = ChromaCollections()
    results: dict[str, Any] = {}
    for name, handle in (("nl2sql", cols.nl2sql), ("planner", cols.planner)):
        results[name] = sweep_collection(
            name, handle,
            ttl_days=int(args.get("ttl_days", 30)),
            min_confidence=float(args.get("min_confidence", 0.2)),
            dry_run=bool(args.get("dry_run", False)),
        )
    return results


def _run_seed_planner(_args: dict) -> dict:
    from services.embeddings_service import EmbeddingsService
    from services.planner_memory import (
        SEED_MODULE_CARDS,
        SEED_WORKFLOW_EXEMPLARS,
        PlannerMemory,
    )
    embeddings = EmbeddingsService()
    memory = PlannerMemory()
    n_cards = 0
    for card in SEED_MODULE_CARDS:
        memory.upsert_module_card(card, embeddings.embed(card.doc_text()))
        n_cards += 1
    n_ex = 0
    for ex in SEED_WORKFLOW_EXEMPLARS:
        memory.upsert_workflow_success(ex, embeddings.embed(ex.doc_text()))
        n_ex += 1
    return {"module_cards": n_cards, "exemplars": n_ex}


_RUNNERS: dict[str, Callable[[dict], Any]] = {
    "official_ingestion": _run_official_ingestion,
    "precompute_v2":      _run_precompute_v2,
    "decay_experience":   _run_decay_experience,
    "seed_planner":       _run_seed_planner,
}


def execute(task: TaskSpec) -> None:
    """Run a single task with logging + exception isolation."""
    runner = _RUNNERS.get(task.kind)
    if runner is None:
        log.error("scheduler.unknown_kind", task=task.name, kind=task.kind)
        return
    started = datetime.now(timezone.utc).isoformat()
    log.info("scheduler.task_start", task=task.name, kind=task.kind,
             started_at=started)
    try:
        result = runner(task.args)
        log.info("scheduler.task_done", task=task.name, kind=task.kind,
                 result=result)
    except Exception as exc:
        log.error("scheduler.task_failed",
                  task=task.name, kind=task.kind,
                  error=str(exc)[:200],
                  traceback=traceback.format_exc()[:1500])


# ── Mode entry points ────────────────────────────────────────────────────────

def cmd_list(tasks: list[TaskSpec]) -> int:
    if not tasks:
        print("(no tasks defined)")
        return 0
    for t in tasks:
        flag_enabled = "+" if t.enabled else "-"
        flag_start = "*" if t.run_on_start else " "
        print(f"  [{flag_enabled}{flag_start}] {t.name:30s} "
              f"kind={t.kind:18s} cron={t.cron or '-'}")
    print()
    print("Legend: + enabled / - disabled, * runs at startup")
    return 0


def cmd_once(tasks: list[TaskSpec], task_name: str) -> int:
    matches = [t for t in tasks if t.name == task_name]
    if not matches:
        log.error("scheduler.task_not_found", task=task_name,
                  known=[t.name for t in tasks])
        return 2
    execute(matches[0])
    return 0


def cmd_bootstrap(tasks: list[TaskSpec]) -> int:
    """Run every enabled task once and exit. Useful for first-deploy bootstrap."""
    runnable = [t for t in tasks if t.enabled and t.run_on_start]
    log.info("scheduler.bootstrap_start", count=len(runnable))
    for t in runnable:
        execute(t)
    log.info("scheduler.bootstrap_done")
    return 0


def cmd_apscheduler(tasks: list[TaskSpec]) -> int:
    """Long-running daemon: run-on-start once, then schedule on cron."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        log.error("scheduler.apscheduler_missing",
                  hint="pip install apscheduler")
        return 3

    enabled = [t for t in tasks if t.enabled]
    if not enabled:
        log.warning("scheduler.no_enabled_tasks")
        return 0

    # 1. Run-on-start phase: every enabled task with run_on_start=true gets
    #    one immediate execution. This is what makes "first deploy works"
    #    without a separate bootstrap step.
    bootstrap_tasks = [t for t in enabled if t.run_on_start]
    if bootstrap_tasks:
        log.info("scheduler.run_on_start_phase",
                 count=len(bootstrap_tasks),
                 names=[t.name for t in bootstrap_tasks])
        for t in bootstrap_tasks:
            execute(t)

    # 2. Cron phase: register each task's cron trigger.
    sched = BlockingScheduler(timezone="UTC")
    for t in enabled:
        if not t.cron:
            log.info("scheduler.task_skipped_no_cron", task=t.name)
            continue
        try:
            trigger = CronTrigger.from_crontab(t.cron)
        except Exception as exc:
            log.error("scheduler.invalid_cron",
                      task=t.name, cron=t.cron, error=str(exc)[:120])
            continue
        sched.add_job(execute, trigger, args=[t],
                      id=t.name, replace_existing=True)
        log.info("scheduler.task_registered",
                 task=t.name, cron=t.cron)

    log.info("scheduler.start", task_count=len(enabled))
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler.stop")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="v2 scheduler - daily cron + first-deploy bootstrap",
    )
    parser.add_argument("--mode", default="apscheduler",
                        choices=["apscheduler", "once", "bootstrap", "list"],
                        help="Default: apscheduler (long-running daemon).")
    parser.add_argument("--task", default=None,
                        help="Required with --mode once (task name).")
    parser.add_argument("--config", default=None,
                        help="Path to scheduler_tasks.yaml (default: "
                             "config/scheduler_tasks.yaml).")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Shortcut for --mode bootstrap.")
    parser.add_argument("--once", action="store_true",
                        help="Shortcut for --mode once.")
    parser.add_argument("--list", action="store_true",
                        help="Shortcut for --mode list.")
    args = parser.parse_args()

    if args.list:
        args.mode = "list"
    elif args.bootstrap:
        args.mode = "bootstrap"
    elif args.once:
        args.mode = "once"

    cfg_path = Path(args.config) if args.config else None
    tasks = load_tasks(cfg_path)
    if not tasks:
        log.error("scheduler.no_tasks_loaded")
        return 1

    if args.mode == "list":
        return cmd_list(tasks)
    if args.mode == "bootstrap":
        return cmd_bootstrap(tasks)
    if args.mode == "once":
        if not args.task:
            parser.error("--mode once requires --task NAME")
        return cmd_once(tasks, args.task)
    return cmd_apscheduler(tasks)


if __name__ == "__main__":
    sys.exit(main())
