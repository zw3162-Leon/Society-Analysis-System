"""
KG Query tools - redesign-2026-05-kg Phase B.

Branch C (Knowledge Graph Query). Cypher-only operations against Kuzu.
The companion module `agents/kg_analytics.py` covers NetworkX-backed
algorithmic analyses (PageRank / Louvain / betweenness).

Five query kinds (Phase B.1 + B.5):
    1. propagation_path  - reply chain between two ACCOUNTS via Post-Post
                           Replied edges, then bridges back to the authors
    2. key_nodes         - top accounts in a topic by post count
                           (Phase B.3 supersedes with PageRank in
                           KGAnalytics.influencer_rank)
    3. topic_correlation - shared-entity strength between two topics
    4. cascade_tree      - full reply tree under a single root post
    5. viral_cascade     - top-K longest / fastest cascades in a topic

Removed in Phase B.6:
    - community_relations  (replaced by KGAnalytics.coordinated_groups,
                            which uses Louvain modularity instead of a
                            naive co-posting count)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import structlog

from models.branch_output import KGEdge, KGNode, KGOutput
from services.kuzu_service import KuzuService

log = structlog.get_logger(__name__)


def _fetch_post_meta(post_ids: set[str]) -> dict[str, dict]:
    """Hydrate a set of Post ids with their text + author from Postgres.

    Best-effort: returns an empty dict if PG is unreachable. Used by
    propagation_path / cascade_tree / viral_cascade to expose
    human-readable content alongside raw graph ids.
    """
    if not post_ids:
        return {}
    try:
        from services.postgres_service import PostgresService
        pg = PostgresService()
        pg.connect()
        with pg.cursor() as cur:
            cur.execute(
                "SELECT post_id, author, text FROM posts_v2 "
                "WHERE post_id = ANY(%s)",
                (list(post_ids),),
            )
            rows = list(cur.fetchall())
    except Exception:
        return {}
    return {
        r["post_id"]: {
            "author": r.get("author") or "",
            "text": r.get("text") or "",
        }
        for r in rows
    }


@dataclass
class KGQueryTool:
    kuzu: Optional[KuzuService] = None

    def __post_init__(self) -> None:
        if self.kuzu is None:
            try:
                self.kuzu = KuzuService()
            except Exception as exc:
                log.warning("kg.kuzu_unavailable", error=str(exc)[:120])
                self.kuzu = None

    # ── 1. propagation_path ──────────────────────────────────────────────────

    def propagation_path(
        self,
        source_account: str,
        target_account: str,
        max_hops: int = 6,
    ) -> KGOutput:
        """Find a reply chain that connects two ACCOUNTS.

        Walks the Post-[:Replied*]->Post backbone and returns the path
        plus the involved authors. Reply edges go child -> parent in our
        schema, so we expand both directions to handle "did A end up
        responding to B" and "did B respond to A" symmetrically.
        """
        t0 = time.monotonic()
        out = KGOutput(
            query_kind="propagation_path",
            target={"source": source_account, "target": target_account,
                     "max_hops": max_hops},
        )
        if not self.kuzu:
            out.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return out

        # Reply edges are directed child -> parent in our schema.
        # We try both directions: target replied (transitively) to source,
        # then source replied to target. Kuzu 0.7 doesn't support
        # `[n IN nodes(p) | ...]` comprehensions, so we project nodes()
        # raw and unpack node ids in Python.
        # Two passes (directed each way) merged in Python to keep the
        # Cypher simple and Kuzu-compatible.
        results = []
        for src_pid_node, dst_pid_node in [("pa", "pb"), ("pb", "pa")]:
            cypher = (
                "MATCH (a:Account {id: $aid})-[:Posted]->(pa:Post), "
                "      (b:Account {id: $bid})-[:Posted]->(pb:Post), "
                f"      path = ({src_pid_node})-[:Replied*1..{max_hops}]"
                f"->({dst_pid_node}) "
                "RETURN nodes(path) AS chain LIMIT 5"
            )
            rows = self.kuzu._safe_execute(  # type: ignore[attr-defined]
                cypher, {"aid": source_account, "bid": target_account},
            ) or []
            results.extend(rows)

        # Collect every post id along every chain so we can fetch
        # human-readable text + author in one shot.
        all_post_ids: list[str] = []
        chains: list[list[str]] = []
        for row in results:
            chain = row.get("chain") or []
            ids = []
            for n in chain:
                pid = (n.get("id") if isinstance(n, dict)
                       else getattr(n, "id", None) or str(n))
                ids.append(str(pid))
            chains.append(ids)
            all_post_ids.extend(ids)

        # Hydrate post text + author from Postgres so the Report Writer
        # gets natural-language context instead of opaque ids.
        post_meta = _fetch_post_meta(set(all_post_ids))

        # Drop chains whose posts cannot be hydrated. Kuzu retains nodes
        # across overwrite imports while Postgres is wiped per subreddit;
        # paths that route through orphan Kuzu posts produce empty
        # author/text strings and force the Writer to invent a narrative.
        # Prefer fully-hydratable chains; if none survive, fall back to
        # the original set rather than returning an empty path.
        hydrated_chains = [
            ids for ids in chains
            if all((post_meta.get(p, {}).get("author") or "").strip()
                   for p in ids)
        ]
        if hydrated_chains:
            chains = hydrated_chains
            log.info("kg.propagation_path.filtered_orphans",
                     kept=len(hydrated_chains),
                     dropped=len(all_post_ids) - sum(len(c) for c in hydrated_chains))

        seen_posts: set[str] = set()
        for ids in chains:
            prev = None
            for pid in ids:
                if pid not in seen_posts:
                    seen_posts.add(pid)
                    meta = post_meta.get(pid, {})
                    text = (meta.get("text") or "").strip()
                    out.nodes.append(KGNode(
                        id=pid, label="Post",
                        properties={
                            "author": meta.get("author") or "",
                            "text": (text[:160]
                                      if len(text) > 160 else text),
                        },
                    ))
                if prev:
                    out.edges.append(KGEdge(
                        source_id=prev, target_id=pid, rel_type="Replied",
                    ))
                prev = pid

        # Always include the two endpoint accounts so the report writer can
        # show "alice -> bob" in prose.
        for acc in (source_account, target_account):
            out.nodes.append(KGNode(id=acc, label="Account"))

        out.metrics["paths_found"] = len(results)
        out.metrics["max_path_length"] = max(
            (len(r.get("chain") or []) for r in results), default=0,
        )
        out.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return out

    # ── 2. key_nodes ─────────────────────────────────────────────────────────

    def key_nodes(self, topic_id: str, top_k: int = 10) -> KGOutput:
        """Top accounts in a topic by raw post count.

        Phase B.3 introduces `KGAnalytics.influencer_rank` which uses
        PageRank for "real" influence. This stays as the cheap fallback.
        """
        t0 = time.monotonic()
        out = KGOutput(query_kind="key_nodes",
                       target={"topic_id": topic_id, "top_k": top_k})
        if not self.kuzu:
            out.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return out

        rows = self.kuzu._safe_execute(  # type: ignore[attr-defined]
            "MATCH (a:Account)-[:Posted]->(p:Post)-[:BelongsToTopic]->(t:Topic {id: $tid}) "
            "RETURN a.id AS account_id, a.username AS username, "
            "count(p) AS post_count "
            "ORDER BY post_count DESC LIMIT $k",
            {"tid": topic_id, "k": top_k},
        ) or []
        for r in rows:
            out.nodes.append(KGNode(
                id=r["account_id"], label="Account",
                properties={"username": r.get("username", ""),
                             "post_count": r.get("post_count", 0)},
            ))
        out.metrics["accounts_in_topic"] = len(rows)
        out.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return out

    # ── 3. topic_correlation ─────────────────────────────────────────────────

    def topic_correlation(
        self, topic_a: str, topic_b: str,
    ) -> KGOutput:
        t0 = time.monotonic()
        out = KGOutput(
            query_kind="topic_correlation",
            target={"topic_a": topic_a, "topic_b": topic_b},
        )
        if not self.kuzu:
            out.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return out

        rows = self.kuzu._safe_execute(  # type: ignore[attr-defined]
            "MATCH (p1:Post)-[:BelongsToTopic]->(t1:Topic {id: $a}), "
            "(p1)-[:HasEntity]->(e:Entity), "
            "(p2:Post)-[:HasEntity]->(e), "
            "(p2)-[:BelongsToTopic]->(t2:Topic {id: $b}) "
            "RETURN DISTINCT e.id AS entity_id, e.name AS name, "
            "e.entity_type AS entity_type",
            {"a": topic_a, "b": topic_b},
        ) or []
        for r in rows:
            out.nodes.append(KGNode(
                id=r["entity_id"], label="Entity",
                properties={"name": r.get("name", ""),
                             "entity_type": r.get("entity_type", "")},
            ))
        out.metrics["shared_entity_count"] = len(out.nodes)
        out.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return out

    # ── 4. cascade_tree ──────────────────────────────────────────────────────

    def cascade_tree(
        self, root_post_id: str, max_depth: int = 10,
    ) -> KGOutput:
        """Recursively expand the reply tree rooted at a single post.

        Returns every reachable Post node (regardless of topic / author)
        plus every Replied edge, so the UI can draw a real cascade.
        """
        t0 = time.monotonic()
        out = KGOutput(
            query_kind="cascade_tree",
            target={"root_post_id": root_post_id, "max_depth": max_depth},
        )
        if not self.kuzu:
            out.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return out

        # Collect every (child, parent) pair that descends from the root
        # via any number of Replied hops up to max_depth. Reply edges go
        # child -> parent, so we follow them in reverse from the root.
        #
        # Kuzu cannot reuse an UNWINDed node value as a node pattern, so keep
        # the root row and descendant rows as separate simple MATCH queries.
        root_rows = self.kuzu._safe_execute(  # type: ignore[attr-defined]
            "MATCH (a:Account)-[:Posted]->(root:Post {id: $rid}) "
            "RETURN root.id AS post_id, '' AS parent_id, a.id AS account_id",
            {"rid": root_post_id},
        ) or []
        if not root_rows:
            root_rows = self.kuzu._safe_execute(  # type: ignore[attr-defined]
                "MATCH (root:Post {id: $rid}) "
                "RETURN root.id AS post_id, '' AS parent_id, '' AS account_id",
                {"rid": root_post_id},
            ) or []
        desc_cypher = (
            "MATCH (root:Post {id: $rid})"
            "<-[:Replied*1.." + str(max_depth) + "]-(p:Post) "
            "OPTIONAL MATCH (p)-[:Replied]->(parent:Post) "
            "OPTIONAL MATCH (a:Account)-[:Posted]->(p) "
            "RETURN p.id AS post_id, parent.id AS parent_id, "
            "       a.id AS account_id"
        )
        desc_rows = self.kuzu._safe_execute(  # type: ignore[attr-defined]
            desc_cypher, {"rid": root_post_id},
        ) or []
        rows = list(root_rows) + list(desc_rows)

        post_authors: dict[str, Optional[str]] = {}
        for r in rows:
            pid = r.get("post_id")
            if pid is None:
                continue
            post_authors[str(pid)] = r.get("account_id")
            parent_id = r.get("parent_id")
            if parent_id and pid != parent_id:
                out.edges.append(KGEdge(
                    source_id=str(pid),
                    target_id=str(parent_id),
                    rel_type="Replied",
                ))

        # Hydrate readable text + author for every post in the cascade.
        post_meta = _fetch_post_meta(set(post_authors))
        for pid, account_id in post_authors.items():
            meta = post_meta.get(pid, {})
            text = (meta.get("text") or "").strip()
            out.nodes.append(KGNode(
                id=pid, label="Post",
                properties={
                    "author_id": account_id or "",
                    "author":    meta.get("author") or account_id or "",
                    "text": (text[:160] if len(text) > 160 else text),
                },
            ))

        out.metrics["cascade_size"] = max(0, len(post_authors) - 1)
        out.metrics["unique_authors"] = len(
            {a for a in post_authors.values() if a}
        )
        out.metrics["depth_hint"] = len(out.edges)
        out.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return out

    # ── 5. viral_cascade ─────────────────────────────────────────────────────

    def viral_cascade(
        self, topic_id: str, top_k: int = 5,
    ) -> KGOutput:
        """Top-K most-amplified cascades inside a topic.

        Ranks root posts by descendant count + unique-author count. Each
        result row becomes a node carrying the cascade metrics so the
        Report Writer can show "this rumour was reposted by 12 accounts
        across a 3-deep reply chain in 4 hours".
        """
        t0 = time.monotonic()
        out = KGOutput(
            query_kind="viral_cascade",
            target={"topic_id": topic_id, "top_k": top_k},
        )
        if not self.kuzu:
            out.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return out

        # Most-amplified posts in this topic, ranked by descendant count.
        # Kuzu does not support `WHERE NOT (var)-[:rel]->(...)` pattern
        # predicates - the match silently returns 0 rows. The previous formulation
        # hard-filtered to "true roots" via that idiom and produced an empty
        # cascade output even on topics with hundreds of reply edges.
        # Drop the strict-root filter: a "viral" cascade is now any post in
        # the topic with cascade_size > 0, regardless of whether the post
        # itself is a reply. This matches what users mean by "most amplified
        # posts" and survives the cross-topic-reply pattern that's common
        # in this dataset (most parent posts of in-topic replies live in
        # other topics).
        cypher = (
            "MATCH (root:Post)-[:BelongsToTopic]->(t:Topic {id: $tid}) "
            "OPTIONAL MATCH (child:Post)-[:Replied*1..10]->(root) "
            "OPTIONAL MATCH (a:Account)-[:Posted]->(child) "
            "WITH root, count(DISTINCT child) AS cascade_size, "
            "     count(DISTINCT a)            AS unique_authors "
            "WHERE cascade_size > 0 "
            "RETURN root.id AS root_id, root.text AS text, "
            "       cascade_size, unique_authors "
            "ORDER BY cascade_size DESC, unique_authors DESC LIMIT $k"
        )
        rows = self.kuzu._safe_execute(cypher,  # type: ignore[attr-defined]
                                        {"tid": topic_id, "k": top_k}) or []

        for r in rows:
            out.nodes.append(KGNode(
                id=str(r["root_id"]), label="Post",
                properties={
                    "text": (r.get("text") or "")[:200],
                    "cascade_size": int(r.get("cascade_size", 0)),
                    "unique_authors": int(r.get("unique_authors", 0)),
                },
            ))

        out.metrics["cascade_count"] = len(rows)
        out.metrics["max_cascade_size"] = max(
            (int(r.get("cascade_size", 0)) for r in rows), default=0,
        )
        out.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return out

    def topic_reply_chains(
        self, topic_id: str, top_k: int = 5, max_depth: int = 5,
    ) -> KGOutput:
        """Return concrete reply-chain nodes/edges for top cascades in a topic.

        `viral_cascade` ranks candidate root posts. This method expands those
        roots into actual Post nodes plus Replied edges so report writing can
        show chain excerpts instead of only cascade scores.
        """
        t0 = time.monotonic()
        out = KGOutput(
            query_kind="reply_chains",
            target={"topic_id": topic_id, "top_k": top_k,
                    "max_depth": max_depth},
        )
        if not topic_id:
            out.target["reason"] = "missing topic_id"
            return out

        ranked = self.viral_cascade(topic_id=topic_id, top_k=top_k)
        if not ranked.nodes:
            out.metrics = {"cascade_count": 0}
            out.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return out

        seen_nodes: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()
        for idx, root in enumerate(ranked.nodes, start=1):
            tree = self.cascade_tree(root.id, max_depth=max_depth)
            cascade_size = root.properties.get("cascade_size", 0)
            unique_authors = root.properties.get("unique_authors", 0)
            for node in tree.nodes:
                if node.id in seen_nodes:
                    continue
                props = dict(node.properties)
                props.update({
                    "root_post_id": root.id,
                    "chain_rank": idx,
                    "root_cascade_size": cascade_size,
                    "root_unique_authors": unique_authors,
                })
                out.nodes.append(KGNode(
                    id=node.id,
                    label=node.label,
                    properties=props,
                ))
                seen_nodes.add(node.id)
            for edge in tree.edges:
                key = (edge.source_id, edge.target_id, edge.rel_type)
                if key in seen_edges:
                    continue
                out.edges.append(edge)
                seen_edges.add(key)

        out.metrics = {
            "cascade_count": len(ranked.nodes),
            "node_count": len(out.nodes),
            "edge_count": len(out.edges),
            "max_cascade_size": ranked.metrics.get("max_cascade_size", 0),
        }
        out.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return out
