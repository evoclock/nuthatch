# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Concept + summary augmentation primitives for the corpus graph.

Pure library functions invoked by `nuthatch.graph.orchestrator` after
the bibliographic graph is built. They are deliberately NOT a CLI:
the augmentation belongs INSIDE the graph build so any re-run of
`nuthatch graph` produces the same fully-augmented graph from the same
sidecars. Treating concept augmentation as a separate post-pass was
brittle - it stranded the augmented state outside the build's idempotent
contract.

Reads (per doc, optional):
  - `<corpus>/.kg/extracted/<doc_id>.concepts.json`

Writes (into the in-memory NetworkX graph the orchestrator hands us):
  - Sets `summary`, `domain` attributes on the document node when a
    sidecar exists.
  - Adds topic / method / named_entity nodes (one per unique concept
    across the corpus, slug-keyed).
  - Adds `mentions` edges from document -> concept node.
  - Optionally adds `shares_summary_with` edges between document pairs
    whose summary embeddings are similar above a threshold.

If no concept sidecars exist, both functions are no-ops; the corpus
just gets a bibliographic-only graph (the prior default).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_")


def load_concept_sidecars(extracted_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all `<doc_id>.concepts.json` sidecars from extracted/.

    Returns a dict keyed by bare doc_id (matches `<doc_id>.md`). Bad
    sidecars (unparseable JSON) are skipped with no error so a single
    bad file does not block the rest of the build.
    """
    out: dict[str, dict[str, Any]] = {}
    if not extracted_dir.is_dir():
        return out
    for cf in sorted(extracted_dir.glob("*.concepts.json")):
        try:
            c = json.loads(cf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        doc_id = c.get("doc_id") or cf.stem.replace(".concepts", "")
        out[doc_id] = c
    return out


def augment_with_concepts(
    g: nx.MultiDiGraph,
    concepts_by_doc: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """Add topic/method/named_entity nodes and `mentions` edges to `g`.

    Idempotent at the corpus level: re-running with the same sidecars
    produces the same node + edge set (slug-keyed concept node ids
    collide deterministically). Sets `summary` + `domain` attributes
    on the source document nodes.

    Returns `(added_nodes, added_edges)`.
    """
    added_nodes = 0
    added_edges = 0
    existing_concept_keys: set[str] = set()

    # Document nodes in the graph are stored as `doc::<doc_id>`. Build
    # a bare-id -> graph-node-id map once for fast lookup.
    doc_nodes_by_bare_id: dict[str, str] = {}
    for node, data in g.nodes(data=True):
        if data.get("node_type") == "document":
            bare = str(node).split("::", 1)[-1]
            doc_nodes_by_bare_id[bare] = str(node)

    for doc_id, concepts in concepts_by_doc.items():
        doc_node = doc_nodes_by_bare_id.get(doc_id)
        if doc_node is None:
            continue
        if concepts.get("summary"):
            g.nodes[doc_node]["summary"] = concepts["summary"]
        if concepts.get("domain"):
            g.nodes[doc_node]["domain"] = concepts["domain"]

        for raw, kind in (
            *((t, "topic") for t in concepts.get("topics", [])),
            *((m, "method") for m in concepts.get("methods", [])),
            *((n, "named_entity") for n in concepts.get("named_entities", [])),
        ):
            slug = _slug(raw)
            if not slug:
                continue
            node_id = f"{kind}::{slug}"
            if node_id not in existing_concept_keys:
                if node_id not in g:
                    g.add_node(
                        node_id,
                        node_type="entity",
                        entity_type=kind,
                        name=raw,
                    )
                    added_nodes += 1
                existing_concept_keys.add(node_id)
            g.add_edge(doc_node, node_id, relation="mentions")
            added_edges += 1
    return added_nodes, added_edges


def add_semantic_edges(
    g: nx.MultiDiGraph,
    concepts_by_doc: dict[str, dict[str, Any]],
    *,
    threshold: float = 0.55,
    embedder: Any | None = None,
) -> int:
    """Add `shares_summary_with` edges between docs with similar summaries.

    Uses BGE-M3 (the same model as the retriever) so the semantic
    space matches the rest of the pipeline. Skips when fewer than 2
    summaries are available. Returns the number of edges added.

    `embedder` is dependency-injected for tests; production passes
    `None` and a fresh `Embedder()` is constructed.
    """
    summaries: list[tuple[str, str]] = [
        (doc_id, c["summary"]) for doc_id, c in concepts_by_doc.items() if c.get("summary")
    ]
    if len(summaries) < 2:
        return 0

    import numpy as np

    if embedder is None:
        from nuthatch.embed.embed import Embedder

        embedder = Embedder()
    texts = [s for _, s in summaries]
    vecs = embedder.encode(texts)
    arr = np.asarray(vecs, dtype=np.float32)
    sim = arr @ arr.T  # BGE-M3 vectors are unit-norm -> dot is cosine

    doc_nodes_by_bare_id: dict[str, str] = {}
    for node, data in g.nodes(data=True):
        if data.get("node_type") == "document":
            bare = str(node).split("::", 1)[-1]
            doc_nodes_by_bare_id[bare] = str(node)

    n_edges = 0
    n_docs = len(summaries)
    for i in range(n_docs):
        di = summaries[i][0]
        u = doc_nodes_by_bare_id.get(di)
        if u is None:
            continue
        for j in range(i + 1, n_docs):
            if sim[i, j] < threshold:
                continue
            dj = summaries[j][0]
            v = doc_nodes_by_bare_id.get(dj)
            if v is None:
                continue
            g.add_edge(
                u,
                v,
                relation="shares_summary_with",
                similarity=float(sim[i, j]),
            )
            n_edges += 1
    return n_edges
