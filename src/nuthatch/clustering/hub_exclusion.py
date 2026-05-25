# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""High-degree node detection + exclusion for cleaner community detection.

Purpose: paper knowledge graphs contain a small number of
    enormously-cited "core" papers (the Darwins, the BLAST papers,
    the Word2Vec papers) that touch every community. Including them
    in the partition pulls many communities together into a single
    blob; excluding them and reattaching to majority-vote community
    afterwards gives a cleaner structural partition.

Inputs at `core_nodes`: a `networkx` graph + percentile cutoff for
    "high degree".

Outputs at `core_nodes`: the list of node IDs whose degree is at or
    above the percentile cutoff.

Pattern derived from kestrel's high-degree-node detection in
`analyze.py`. nuthatch's implementation uses `core_nodes` as the
neutral function and constant name (the kestrel name is avoided
per the project's naming policy); the algorithm is straightforward
percentile-rank on node degree, which is the same conceptual move
the source pattern makes.

Reattachment of excluded nodes is by majority-vote neighbour
community: after the partitioner runs on the graph-minus-core, each
core node is reattached to the community that holds the largest
share of its neighbours. Ties break by community ID (deterministic).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import networkx as nx


def core_nodes(
    g: nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph,
    *,
    percentile: float = 99.0,
) -> list[str]:
    """Return node IDs whose total degree is at or above `percentile`.

    `percentile` is 0-100. The default 99.0 picks the top ~1% of
    nodes by connectivity; tune per-corpus via config.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if g.number_of_nodes() == 0:
        return []
    # Use total degree (in + out for directed). For multigraphs each
    # parallel edge counts; that matches the "loud" intuition.
    degrees = {
        node: int(g.degree(node))  # type: ignore[arg-type]
        for node in g.nodes()
    }
    sorted_degrees = sorted(degrees.values())
    # Percentile cutoff via linear interpolation rank.
    rank = int(round((percentile / 100.0) * (len(sorted_degrees) - 1)))
    cutoff = sorted_degrees[rank]
    return [node for node, d in degrees.items() if d >= cutoff]


def exclude_core_nodes(
    g: nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph,
    *,
    percentile: float = 99.0,
) -> tuple[nx.Graph, list[str]]:
    """Return (graph_without_core_nodes, excluded_node_ids).

    The returned graph is a subgraph view; mutating it does not
    affect the source `g`. The partitioner runs on the subgraph;
    reattachment happens via `reattach_by_majority_neighbour`.
    """
    core = set(core_nodes(g, percentile=percentile))
    keep = [n for n in g.nodes() if n not in core]
    # Use `subgraph` then `copy()` so downstream mutation is safe.
    sub = g.subgraph(keep).copy()
    # If the source is a multigraph or directed, the subgraph
    # preserves that — return as the broad nx.Graph type for the
    # caller to handle uniformly.
    return sub, list(core)


def reattach_by_majority_neighbour(
    g: nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph,
    partition: dict[str, int],
    excluded_nodes: Iterable[str],
) -> dict[str, int]:
    """Add the excluded core nodes back to the partition by majority vote.

    For each excluded node, count its neighbours' communities and
    assign the most common. Ties break deterministically by lower
    community ID. Nodes with no neighbours land in a new singleton
    community (max(partition) + 1, monotonically increasing).
    """
    partition = dict(partition)
    next_community = max(partition.values(), default=-1) + 1
    for node in excluded_nodes:
        if node not in g:
            continue
        neighbours = (
            list(g.successors(node)) + list(g.predecessors(node))  # type: ignore[union-attr]
            if g.is_directed()
            else list(g.neighbors(node))
        )
        votes = Counter(
            partition[n] for n in neighbours if n in partition
        )
        if votes:
            top_count = max(votes.values())
            tied = sorted(c for c, count in votes.items() if count == top_count)
            partition[node] = tied[0]
        else:
            partition[node] = next_community
            next_community += 1
    return partition
