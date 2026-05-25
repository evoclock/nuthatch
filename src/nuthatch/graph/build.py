# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Graph construction from per-paper entities and edges.

Purpose: assemble per-paper `ExtractedEntity` sets + per-paper edge
    lists into a single `networkx.MultiDiGraph` for the corpus.
    `MultiDiGraph` (not `Graph`) so the same source/target pair can
    carry several typed relations (cites + co-mentions + same-author
    chain).

Inputs: a stream of `PaperContribution(paper_node_id, metadata,
    entities, body_markdown)` records — typically produced by the
    ingest orchestrator after a successful schema-validated extract.

Outputs: `nx.MultiDiGraph` with paper nodes, entity nodes, and
    edges from paper -> author / topic / citation, plus
    entity-entity co-mention edges within a paper.

Pattern reused from kestrel's `build.py` (the three-layer node
deduplication: within-source, between-source, and explicit semantic
merge). nuthatch's implementation is a fresh write adapted for
papers (not source-code ASTs): one paper is the unit instead of
one source file, and `add_node` is idempotent at the corpus level
because entity keys are normalised slugs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from nuthatch.graph.edges import Confidence, Edge
from nuthatch.graph.entities import (
    EntityExtractor,
    EntityType,
    ExtractedEntity,
    deduplicate_entities,
)


@dataclass(frozen=True)
class PaperContribution:
    """One paper's worth of input to `build_graph`."""

    paper_node_id: str
    metadata: dict[str, Any]
    body_markdown: str
    entities: list[ExtractedEntity] = field(default_factory=list)


def _paper_attrs(meta: dict[str, Any]) -> dict[str, Any]:
    """Attributes attached to a paper node in the graph."""
    out: dict[str, Any] = {"node_type": "paper"}
    for key in ("title", "year", "doi", "arxiv_id", "doc_id"):
        if meta.get(key):
            out[key] = meta[key]
    return out


def _entity_attrs(entity: ExtractedEntity) -> dict[str, Any]:
    return {
        "node_type": "entity",
        "entity_type": str(entity.type),
        "name": entity.name,
    }


def build_graph(
    contributions: Iterable[PaperContribution],
    *,
    extractor: EntityExtractor | None = None,
) -> nx.MultiDiGraph:
    """Build the corpus graph from per-paper contributions.

    For each paper:
      1. Add the paper node (idempotent on `paper_node_id`).
      2. Run the entity extractor over the body+metadata if
         `entities` was not pre-populated.
      3. Dedup entities within the paper.
      4. Add entity nodes (idempotent on `entity.key` across papers).
      5. Emit typed edges: paper -> author, paper -> topic,
         paper -> citation, with confidence labels.
      6. Emit entity co-mention edges within the paper for entity
         pairs that share the same paper (INFERRED confidence).
    """
    extractor = extractor or EntityExtractor()
    g: nx.MultiDiGraph = nx.MultiDiGraph()

    for c in contributions:
        g.add_node(c.paper_node_id, **_paper_attrs(c.metadata))

        entities = c.entities or extractor.extract(
            markdown=c.body_markdown, metadata=c.metadata
        )
        entities = deduplicate_entities(entities)

        for e in entities:
            if e.key not in g.nodes:
                g.add_node(e.key, **_entity_attrs(e))
            relation = _paper_to_entity_relation(e.type)
            edge = Edge(
                source=c.paper_node_id,
                target=e.key,
                relation=relation,
                confidence=e.confidence,
                score=e.score,
                provenance=dict(e.provenance),
            )
            g.add_edge(edge.source, edge.target, **edge.as_attrs())

        # Co-mention edges between entity pairs within this paper.
        # INFERRED confidence: co-mention is a heuristic signal, not
        # a direct lift from the source text.
        keys = sorted({e.key for e in entities})
        for i, src in enumerate(keys):
            for tgt in keys[i + 1 :]:
                co = Edge(
                    source=src,
                    target=tgt,
                    relation="co_mentioned_in",
                    confidence=Confidence.INFERRED,
                    provenance={"paper_node_id": c.paper_node_id},
                )
                g.add_edge(co.source, co.target, **co.as_attrs())

    return g


def _paper_to_entity_relation(entity_type: EntityType) -> str:
    if entity_type is EntityType.AUTHOR:
        return "authored_by"
    if entity_type is EntityType.CITATION:
        return "cites"
    if entity_type is EntityType.TOPIC:
        return "about"
    return f"mentions_{entity_type.value}"
