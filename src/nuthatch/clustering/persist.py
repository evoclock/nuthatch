# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Persist clustering results to the corpus's `.kg/` tree.

Purpose: emit `<corpus>/.kg/communities.json` so downstream consumers
    (the render layer's card frontmatter, the MCP server's
    community-aware retrieval tools) can look up a doc's community
    membership, the nested hierarchy it sits in, the per-community
    core nodes, and the per-community semantic centroid without
    rerunning clustering.

Inputs: a `CorpusLayout`, the `ClusteringResponse` from a backend run,
    the rendered NetworkX graph (so we can compute per-community
    core nodes), and optional centroid vectors fetched from the
    embedding store.

Outputs:
    `<corpus>/.kg/communities.json` (always)
    `<corpus>/.kg/community_centroids.npy` (when centroids supplied)

Rationale: Leiden / Louvain partition results are flat; SBM via
    graph-tool produces a nested chain (see `backends/sbm.py`). Both
    are routed through this module to a single on-disk shape so the
    MCP `community_*` tools can serve either backend without
    backend-aware branching at query time. The MCP-side benefit is
    documented in `docs/Design_Decisions.md` under "Community-aware
    retrieval".

Assumptions: the orchestrator (`cli._cmd_cluster`) has already run a
    backend and is calling this module synchronously; concurrent
    writes are not supported. The on-disk format is JSON for
    inspectability; centroids land in a sibling `.npy` so the JSON
    stays text-only.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from nuthatch.clustering.hub_exclusion import core_nodes
from nuthatch.clustering.protocol import ClusteringResponse
from nuthatch.corpus.layout import CorpusLayout

# Filename of the persisted index. Stable so external tooling can
# read it without consulting our import surface.
COMMUNITIES_INDEX_FILENAME: str = "communities.json"
COMMUNITY_CENTROIDS_FILENAME: str = "community_centroids.npy"

# JSON schema version. Bump when adding required fields so consumers
# can detect format drift gracefully.
COMMUNITIES_SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class CommunityIndex:
    """In-memory view of `<corpus>/.kg/communities.json`.

    Loaded by the MCP server's community-aware tools and by the
    render layer when it injects `community_id` into card
    frontmatter. Centroids, when present, are loaded from the
    sibling `.npy` file via `numpy.load`.
    """

    # Schema version of the persisted file.
    schema_version: int
    # Leaf-level partition: doc_id -> community_id (int).
    flat: dict[str, int]
    # Nested hierarchy: doc_id -> [b0, b1, b2, ...]. Same as
    # `[flat[doc_id]]` for flat backends; deeper for SBM nested.
    hierarchy: dict[str, list[int]]
    # Inverse view at the leaf level: community_id -> [doc_id, ...].
    members: dict[int, list[str]]
    # Per-community high-degree nodes within that community's
    # subgraph. Empty list if the community is too small to score.
    core_nodes: dict[int, list[str]]
    # Per-community human-readable label. Generated from member
    # titles at render time; absent here unless the caller supplies.
    labels: dict[int, str]
    # Backend metadata for telemetry.
    backend: str
    rigor: str
    n_levels: int
    runtime_seconds: float

    def community_for(self, doc_id: str) -> int | None:
        return self.flat.get(doc_id)

    def hierarchy_for(self, doc_id: str) -> list[int]:
        return self.hierarchy.get(doc_id, [])

    def members_of(self, community_id: int) -> list[str]:
        return self.members.get(community_id, [])

    def core_nodes_of(self, community_id: int) -> list[str]:
        return self.core_nodes.get(community_id, [])


def _normalise_hierarchy(
    response: ClusteringResponse,
) -> dict[str, list[int]]:
    """Return a doc_id -> [level0, level1, ...] hierarchy mapping.

    For flat backends (Leiden / Louvain, `response.hierarchy is None`)
    this is `[partition[doc_id]]` — a single-element list. For SBM
    nested backends we concatenate per-level dicts into per-node
    paths. The output is JSON-serialisable.
    """
    if response.hierarchy is None:
        return {doc_id: [int(cid)] for doc_id, cid in response.partition.items()}
    n_levels = len(response.hierarchy)
    out: dict[str, list[int]] = {}
    for doc_id in response.partition:
        path: list[int] = []
        for level in range(n_levels):
            level_map = response.hierarchy[level]
            if doc_id in level_map:
                path.append(int(level_map[doc_id]))
        out[doc_id] = path
    return out


def _members_by_community(partition: Mapping[str, int]) -> dict[int, list[str]]:
    """Inverse view: community_id -> sorted list of doc_ids."""
    inverse: dict[int, list[str]] = defaultdict(list)
    for doc_id, community_id in partition.items():
        inverse[int(community_id)].append(doc_id)
    return {cid: sorted(docs) for cid, docs in inverse.items()}


def _per_community_core_nodes(
    g: nx.Graph,
    members: Mapping[int, Iterable[str]],
    *,
    percentile: float = 75.0,
) -> dict[int, list[str]]:
    """Compute high-degree nodes WITHIN each community's induced subgraph.

    Graphify exposes a global `god_nodes` tool that surfaces the
    top-degree hubs across the whole graph; nuthatch scopes the same
    concept per-community so an agent can ask "what are the key
    members of community X" without re-walking the global graph.

    Returns at most `ceil(n_members * (1 - percentile/100))` IDs per
    community, sorted by descending degree.
    """
    out: dict[int, list[str]] = {}
    for cid, doc_ids in members.items():
        member_set = {d for d in doc_ids if d in g}
        if len(member_set) < 3:
            # Too small to bother scoring; the community itself is
            # the "key" and there is no internal hierarchy to surface.
            out[cid] = []
            continue
        sub = g.subgraph(member_set)
        try:
            hubs = core_nodes(sub, percentile=percentile)
        except ValueError:
            hubs = []
        # Sort by degree descending so the brief tool can show the
        # top-N consistently across runs.
        hubs.sort(key=lambda n: sub.degree(n), reverse=True)
        out[cid] = hubs
    return out


def _labels_from_members(
    members: Mapping[int, Iterable[str]],
    *,
    paper_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    max_words: int = 6,
) -> dict[int, str]:
    """Generate a human-readable label per community from member titles.

    Heuristic: take the most common content words across member
    titles (excluding stop-words and short tokens). Falls back to
    the doc_id of the first member when no titles are available.
    """
    stop = {
        "the", "a", "an", "and", "or", "of", "for", "in", "on", "with",
        "to", "from", "via", "by", "using", "is", "are",
        "be", "been", "being", "we", "our", "their", "this", "that",
        "these", "those", "approach", "method", "study", "paper",
        "analysis", "based", "novel", "new", "towards", "toward",
    }
    out: dict[int, str] = {}
    for cid, doc_ids in members.items():
        titles: list[str] = []
        for d in doc_ids:
            if paper_metadata and d in paper_metadata:
                t = paper_metadata[d].get("title")
                if t:
                    titles.append(str(t))
        if not titles:
            out[cid] = next(iter(doc_ids), f"community_{cid}")
            continue
        words: list[str] = []
        for title in titles:
            for w in title.lower().split():
                cleaned = "".join(ch for ch in w if ch.isalnum() or ch == "-")
                if len(cleaned) >= 4 and cleaned not in stop and not cleaned.isdigit():
                    words.append(cleaned)
        top = [w for w, _ in Counter(words).most_common(max_words)]
        out[cid] = " ".join(top) if top else next(iter(doc_ids), f"community_{cid}")
    return out


def persist_communities(
    *,
    layout: CorpusLayout,
    response: ClusteringResponse,
    graph: nx.Graph,
    paper_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    centroids: Mapping[int, list[float]] | None = None,
    core_nodes_percentile: float = 75.0,
) -> Path:
    """Write the community index to `<corpus>/.kg/communities.json`.

    When `centroids` is provided, also write
    `<corpus>/.kg/community_centroids.npy` (a numpy `.npy` file: a
    2-D float32 array, row i = centroid for community_id at
    `centroids_index[i]`). Centroids power the `community_search`
    MCP tool (semantic search at the community level).

    Returns the path to the written JSON index.
    """
    partition = dict(response.partition)
    hierarchy = _normalise_hierarchy(response)
    members = _members_by_community(partition)
    cores = _per_community_core_nodes(graph, members, percentile=core_nodes_percentile)
    labels = _labels_from_members(members, paper_metadata=paper_metadata)

    payload: dict[str, Any] = {
        "schema_version": COMMUNITIES_SCHEMA_VERSION,
        "backend": response.backend_used,
        "rigor": response.rigor_used.value,
        "runtime_seconds": float(response.runtime_seconds),
        "n_levels": len(response.hierarchy) if response.hierarchy else 1,
        "notes": response.notes,
        "flat": {doc_id: int(cid) for doc_id, cid in partition.items()},
        "hierarchy": hierarchy,
        "members": {str(cid): docs for cid, docs in members.items()},
        "core_nodes": {str(cid): nodes for cid, nodes in cores.items()},
        "labels": {str(cid): label for cid, label in labels.items()},
    }
    if centroids:
        # Save vectors separately as .npy so JSON stays text-only.
        import numpy as np

        centroids_dir = layout.kg
        centroids_dir.mkdir(parents=True, exist_ok=True)
        cids_sorted = sorted(centroids.keys())
        matrix = np.array([centroids[c] for c in cids_sorted], dtype=np.float32)
        centroids_path = centroids_dir / COMMUNITY_CENTROIDS_FILENAME
        np.save(centroids_path, matrix)
        payload["centroids"] = {
            "path": COMMUNITY_CENTROIDS_FILENAME,
            "shape": list(matrix.shape),
            "dtype": "float32",
            "community_ids": [int(c) for c in cids_sorted],
        }

    layout.kg.mkdir(parents=True, exist_ok=True)
    index_path = layout.kg / COMMUNITIES_INDEX_FILENAME
    index_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return index_path


def load_community_index(layout: CorpusLayout) -> CommunityIndex | None:
    """Read `<corpus>/.kg/communities.json` into a `CommunityIndex`.

    Returns None when no index has been written (cluster stage has
    not run, or backend failed). Consumers (MCP server, render
    layer) call this defensively so the absence of clustering is a
    soft-degraded behaviour, not an error.
    """
    index_path = layout.kg / COMMUNITIES_INDEX_FILENAME
    if not index_path.is_file():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return CommunityIndex(
        schema_version=int(data.get("schema_version", 0)),
        flat={k: int(v) for k, v in data.get("flat", {}).items()},
        hierarchy={k: [int(x) for x in v] for k, v in data.get("hierarchy", {}).items()},
        members={int(k): list(v) for k, v in data.get("members", {}).items()},
        core_nodes={int(k): list(v) for k, v in data.get("core_nodes", {}).items()},
        labels={int(k): str(v) for k, v in data.get("labels", {}).items()},
        backend=str(data.get("backend", "")),
        rigor=str(data.get("rigor", "")),
        n_levels=int(data.get("n_levels", 1)),
        runtime_seconds=float(data.get("runtime_seconds", 0.0)),
    )


def load_community_centroids(
    layout: CorpusLayout,
) -> tuple[Any, list[int]] | None:
    """Load the centroids matrix + the community_id order, or None.

    Returns `(matrix, community_ids)` where `matrix[i]` is the
    centroid for `community_ids[i]`. Used by `community_search`
    (semantic match) at MCP query time.
    """
    index_path = layout.kg / COMMUNITIES_INDEX_FILENAME
    centroids_path = layout.kg / COMMUNITY_CENTROIDS_FILENAME
    if not index_path.is_file() or not centroids_path.is_file():
        return None
    try:
        import numpy as np

        data = json.loads(index_path.read_text(encoding="utf-8"))
        meta = data.get("centroids")
        if not meta:
            return None
        cids: list[int] = [int(c) for c in meta.get("community_ids", [])]
        matrix = np.load(centroids_path)
        if matrix.shape[0] != len(cids):
            return None
        return matrix, cids
    except (json.JSONDecodeError, OSError, ValueError):
        return None
