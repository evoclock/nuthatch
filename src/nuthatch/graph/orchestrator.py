# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Corpus-wide graph build orchestrator.

Walks `<corpus>/.kg/extracted/*.md` (written by `IngestOrchestrator`
after schema validation), runs entity extraction per doc, calls
`build_graph()` to assemble the NetworkX MultiDiGraph, and persists
it to `<corpus>/.kg/graph/graph.json`.

The graph captures:

- Document nodes (one per ingested paper / patent / internal doc),
  with frontmatter fields as node attrs (title, year, topics,
  status, relevance, last_touched, half_life_days).
- Entity nodes (authors, citations, topics) extracted from doc
  metadata + body.
- Edges: `authored_by`, `cites`, `co_mentioned_in` between docs
  and entities.

Always rebuilds from scratch (deterministic, no incremental
need — entity extraction is cheap, runtime scales linearly with
doc count). Idempotent: same inputs produce the same graph.json.

The graph is the substrate the clustering step (`nuthatch cluster`)
fits communities over, and the render step (`nuthatch render`)
reads community membership from. The decay step (`nuthatch decay`)
also reads + writes graph state for the per-node `relevance`
attribute.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nuthatch.corpus.layout import CorpusLayout
from nuthatch.graph.build import DocumentContribution, build_graph
from nuthatch.graph.entities import EntityExtractor
from nuthatch.graph.io import save_graph

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphBuildResult:
    """Aggregate outcome of `build_graph_for_corpus`."""

    n_docs: int
    n_nodes: int
    n_edges: int
    graph_path: Path


def build_graph_for_corpus(
    layout: CorpusLayout,
    *,
    extractor: EntityExtractor | None = None,
) -> GraphBuildResult:
    """Build the corpus graph from `<corpus>/.kg/extracted/*.md`.

    `extractor` is dependency-injected so tests can substitute a
    deterministic fake; production uses the default
    `EntityExtractor` which runs regex-based author / topic /
    citation extraction.
    """
    extracted_dir = layout.extracted_dir
    extractor = extractor or EntityExtractor()

    contributions: list[DocumentContribution] = []
    if extracted_dir.is_dir():
        for md_path in sorted(extracted_dir.glob("*.md")):
            doc_id = md_path.stem
            meta_path = extracted_dir / f"{doc_id}.meta.json"
            meta = _load_meta_sidecar(meta_path)
            metadata = meta.get("metadata", {}) if isinstance(meta, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            body = md_path.read_text(encoding="utf-8")
            contributions.append(
                DocumentContribution(
                    doc_node_id=f"doc::{doc_id}",
                    metadata=metadata,
                    body_markdown=body,
                    entities=[],
                )
            )

    g = build_graph(contributions, extractor=extractor)

    graph_path = layout.kg / "graph" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(g, graph_path)

    return GraphBuildResult(
        n_docs=len(contributions),
        n_nodes=g.number_of_nodes(),
        n_edges=g.number_of_edges(),
        graph_path=graph_path,
    )


def _load_meta_sidecar(meta_path: Path) -> dict[str, Any]:
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
