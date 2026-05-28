# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Bipartite projection of the augmented corpus graph onto its document nodes.

The augmented graph carries documents (~3%) and entities (~97% of nodes:
authors, citations, topics, methods, named entities). Most edges are
entity-mediated (`co_mentioned_in`). Clustering algorithms like SBM and
Leiden, run directly on this shape, find tight blocks of co-occurring
ENTITIES — not blocks of papers sharing concepts. The output is dominated
by entity-only communities, useless for "papers grouped by theme".

This module projects the augmented graph onto a doc-doc weighted graph
where docs that share entities get pulled together. The projection
preserves the conceptual signal carried by entity co-mentions while
making documents the unit of clustering.

Edge weight semantics (sum of contributions):
  - Shared entity neighbours: for each entity node touched by both
    docs, +1.0 to the edge weight. Captures the bipartite
    co-occurrence signal directly.
  - Direct `cites` edges: +1.0 each. Citation links between two
    documents in the corpus carry a known relationship.
  - Direct `shares_summary_with` edges: weight = stored `similarity`
    value (BGE-M3 cosine; typically 0.55-0.95). Semantic-similarity
    edges between document summaries.

Output: undirected `nx.Graph` (weighted). Document attributes from the
source graph are preserved on the projected nodes so downstream
consumers (cluster persistence, render) still see titles + summaries.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx

# Weight multipliers applied during projection. Empirically tuned to
# stop the entity-share noise floor from drowning out the two signals
# that actually track topic similarity:
#   - `shares_summary_with` reflects BGE-M3 cosine similarity between
#     document summaries. This is the most topic-bearing signal in
#     the corpus; we multiply the raw similarity by this factor so
#     a typical 0.7-similarity pair contributes ~7.0 to the edge
#     weight (vs the ~1.0 baseline of a single shared entity).
#   - `cites` reflects an asserted directed relationship between
#     two papers. Stronger than entity co-occurrence but weaker than
#     summary similarity; this factor places its unit weight in the
#     middle.
#   - shared-entity neighbours stay at +1.0 each. They are kept (not
#     dropped) because they still carry weak topic signal in dense
#     subfields; they just shouldn't drown out the stronger signals.
_SHARES_SUMMARY_WEIGHT_FACTOR: float = 10.0
_CITES_WEIGHT: float = 5.0


def project_to_doc_doc(g: nx.MultiDiGraph) -> nx.Graph:
    """Project a document-entity bipartite graph onto its document nodes.

    Returns a weighted undirected graph with one node per document
    node in `g`. Documents that share entity neighbours and/or have
    direct doc-doc edges (cites, shares_summary_with) get edges
    proportional to that shared signal.

    Isolated documents (zero entity neighbours, zero doc-doc edges)
    are preserved as nodes so the partition's `n_assigned` count
    matches `len(doc_nodes)` — clustering algorithms place them as
    singletons rather than dropping them.
    """
    proj: nx.Graph = nx.Graph()

    # Collect document nodes; preserve attributes onto the projection
    # so render/persist still have access to titles, summaries, etc.
    doc_node_ids: list[str] = []
    for node, data in g.nodes(data=True):
        if data.get("node_type") == "document":
            doc_node_ids.append(str(node))
            proj.add_node(str(node), **dict(data.items()))

    if len(doc_node_ids) < 2:
        return proj

    doc_set: set[str] = set(doc_node_ids)

    # Build entity -> set-of-docs map. Traverse both directions because
    # the source graph is a MultiDiGraph; `mentions` and similar edges
    # are directed doc -> entity, but `co_mentioned_in` runs entity ->
    # entity and we want every entity that ANY doc touches.
    entity_to_docs: dict[str, set[str]] = {}
    for doc in doc_node_ids:
        for nbr in list(g.successors(doc)) + list(g.predecessors(doc)):
            nbr_data = g.nodes[nbr]
            if nbr_data.get("node_type") == "entity":
                entity_to_docs.setdefault(str(nbr), set()).add(doc)

    # Each entity touched by k docs contributes (k choose 2) edges; for
    # each such pair, +1.0 to weight. Then add direct doc-doc edges
    # on top with their relation-specific weights.
    edge_weights: dict[tuple[str, str], float] = {}
    for docs in entity_to_docs.values():
        if len(docs) < 2:
            continue
        for a, b in combinations(sorted(docs), 2):
            edge_weights[(a, b)] = edge_weights.get((a, b), 0.0) + 1.0

    for u, v, data in g.edges(data=True):
        u_str, v_str = str(u), str(v)
        if u_str not in doc_set or v_str not in doc_set:
            continue
        if u_str == v_str:
            continue
        relation = data.get("relation", "")
        if relation == "shares_summary_with":
            # Stored similarity in [0, 1]; scaled up so a 0.7 cosine
            # match contributes ~7.0 (vs the ~1.0 baseline of one
            # shared entity). Without this boost the semantic signal
            # drowns under entity-co-occurrence noise.
            weight = float(data.get("similarity", 1.0)) * _SHARES_SUMMARY_WEIGHT_FACTOR
        elif relation == "cites":
            # Asserted directed relationship between papers. Stronger
            # than entity co-occurrence; weaker than summary
            # similarity above.
            weight = _CITES_WEIGHT
        else:
            # Other doc-doc relations (unlikely in current schema)
            # default to unit weight, same as a single entity-share.
            weight = 1.0
        a, b = sorted([u_str, v_str])
        edge_weights[(a, b)] = edge_weights.get((a, b), 0.0) + weight

    for (a, b), w in edge_weights.items():
        proj.add_edge(a, b, weight=float(w))

    return proj
