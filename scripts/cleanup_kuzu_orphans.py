"""
cleanup_kuzu_orphans.py

`_delete_reddit_rows` (Overwrite import) only deletes from Postgres. Kuzu
is upsert-only, so old Account/Post/Topic/Entity nodes from prior runs
linger forever. Multi-hop reply paths can route through these orphan
nodes; `_fetch_post_meta` then can't hydrate the post text/author and
the Report Writer is forced to narrate opaque post-ids.

This script aligns Kuzu with Postgres without re-running the pipeline:
  1. Read all Account.id / Post.id / Topic.id / Entity.id from Kuzu.
  2. Read the canonical id sets from posts_v2 / topics_v2 / entities_v2.
  3. Diff → DETACH DELETE the orphans in small batches.

Idempotent and side-effect-free on a healthy graph (zero deletes).

Usage (inside the `api` container):
    python -m scripts.cleanup_kuzu_orphans
    python -m scripts.cleanup_kuzu_orphans --dry-run
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable

import structlog

from services.kuzu_service import KuzuService
from services.postgres_service import PostgresService

log = structlog.get_logger(__name__)


_BATCH = 500


def _kuzu_ids(kuzu: KuzuService, label: str) -> set[str]:
    rows = kuzu._safe_execute(
        f"MATCH (x:{label}) RETURN x.id AS id", {},
    ) or []
    return {str(r["id"]) for r in rows if r.get("id") is not None}


def _pg_post_ids(pg: PostgresService) -> set[str]:
    with pg.cursor() as cur:
        cur.execute("SELECT post_id FROM posts_v2")
        return {r["post_id"] for r in cur.fetchall()}


def _pg_authors(pg: PostgresService) -> set[str]:
    with pg.cursor() as cur:
        cur.execute("SELECT DISTINCT author FROM posts_v2 WHERE author IS NOT NULL")
        return {r["author"] for r in cur.fetchall()}


def _pg_topic_ids(pg: PostgresService) -> set[str]:
    with pg.cursor() as cur:
        cur.execute("SELECT topic_id FROM topics_v2")
        return {r["topic_id"] for r in cur.fetchall()}


def _pg_entity_ids(pg: PostgresService) -> set[str]:
    with pg.cursor() as cur:
        cur.execute("SELECT entity_id FROM entities_v2")
        return {r["entity_id"] for r in cur.fetchall()}


def _detach_delete(
    kuzu: KuzuService, label: str, ids: Iterable[str], dry_run: bool,
) -> int:
    ids = list(ids)
    if not ids:
        return 0
    if dry_run:
        log.info("cleanup.dry_run", label=label, would_delete=len(ids),
                 sample=ids[:5])
        return len(ids)
    deleted = 0
    for start in range(0, len(ids), _BATCH):
        chunk = ids[start:start + _BATCH]
        try:
            kuzu._safe_execute(
                f"MATCH (x:{label}) WHERE x.id IN $ids DETACH DELETE x",
                {"ids": chunk},
            )
            deleted += len(chunk)
        except Exception as exc:
            log.error("cleanup.delete_failed", label=label,
                      batch_start=start, error=str(exc)[:200])
    log.info("cleanup.deleted", label=label, count=deleted)
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="report orphan counts but make no changes")
    args = parser.parse_args()

    pg = PostgresService()
    pg.connect()
    kuzu = KuzuService(read_only=False)

    canon_posts = _pg_post_ids(pg)
    canon_authors = _pg_authors(pg)
    canon_topics = _pg_topic_ids(pg)
    canon_entities = _pg_entity_ids(pg)
    log.info("cleanup.canonical_counts",
             posts=len(canon_posts),
             authors=len(canon_authors),
             topics=len(canon_topics),
             entities=len(canon_entities))

    kuzu_posts = _kuzu_ids(kuzu, "Post")
    kuzu_accounts = _kuzu_ids(kuzu, "Account")
    kuzu_topics = _kuzu_ids(kuzu, "Topic")
    kuzu_entities = _kuzu_ids(kuzu, "Entity")
    log.info("cleanup.kuzu_counts",
             posts=len(kuzu_posts),
             accounts=len(kuzu_accounts),
             topics=len(kuzu_topics),
             entities=len(kuzu_entities))

    orphan_posts = kuzu_posts - canon_posts
    orphan_accounts = kuzu_accounts - canon_authors
    orphan_topics = kuzu_topics - canon_topics
    orphan_entities = kuzu_entities - canon_entities
    log.info("cleanup.orphan_counts",
             posts=len(orphan_posts),
             accounts=len(orphan_accounts),
             topics=len(orphan_topics),
             entities=len(orphan_entities))

    # Order matters when DETACH DELETE traverses rels; doing posts first
    # keeps Account/Topic deletions from cascading through edges that
    # point at posts we're about to delete anyway.
    _detach_delete(kuzu, "Post", orphan_posts, args.dry_run)
    _detach_delete(kuzu, "Account", orphan_accounts, args.dry_run)
    _detach_delete(kuzu, "Topic", orphan_topics, args.dry_run)
    _detach_delete(kuzu, "Entity", orphan_entities, args.dry_run)

    log.info("cleanup.done", dry_run=args.dry_run)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
