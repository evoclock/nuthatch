# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Graph construction from per-document entities and edges.

Purpose: assemble per-document `ExtractedEntity` sets + per-document
    edge lists into a single `networkx.MultiDiGraph` for the corpus.
    `MultiDiGraph` (not `Graph`) so the same source/target pair can
    carry several typed relations (cites + co-mentions + same-author
    chain).

Inputs: a stream of `DocumentContribution(doc_node_id, metadata,
    entities, body_markdown)` records — typically produced by the
    ingest orchestrator after a successful schema-validated extract.

Outputs: `nx.MultiDiGraph` with document nodes, entity nodes, and
    edges from document -> author / topic / citation, plus
    entity-entity co-mention edges within a document.

nuthatch is document-agnostic: papers, patents, technical reports,
internal docs, notes — anything text-based with a SchemaProfile.
Field names and node-type strings use generic terms (`document`,
`doc_node_id`) rather than paper-specific ones.

Pattern adapted from a prior knowledge-graph implementation; original lived in `build.py` (the three-layer node
deduplication: within-source, between-source, and explicit semantic
merge). nuthatch's implementation is a fresh write adapted for
documents (not source-code ASTs): one document is the unit, and
`add_node` is idempotent at the corpus level because entity keys
are normalised slugs.
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
class DocumentContribution:
    """One document's worth of input to `build_graph`.

    A "document" is whatever the active SchemaProfile says it is —
    a paper, a patent, an internal memo, a book chapter. The graph
    builder does not care about the document type beyond storing
    it as the `node_type` attribute on the resulting node.
    """

    doc_node_id: str
    metadata: dict[str, Any]
    body_markdown: str
    entities: list[ExtractedEntity] = field(default_factory=list)


def _document_attrs(meta: dict[str, Any]) -> dict[str, Any]:
    """Attributes attached to a document node in the graph."""
    out: dict[str, Any] = {"node_type": "document"}
    for key in ("title", "year", "doi", "arxiv_id", "patent_number", "doc_id", "type"):
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
    contributions: Iterable[DocumentContribution],
    *,
    extractor: EntityExtractor | None = None,
) -> nx.MultiDiGraph:
    """Build the corpus graph from per-document contributions.

    For each document:
      1. Add the document node (idempotent on `doc_node_id`).
      2. Run the entity extractor over the body+metadata if
         `entities` was not pre-populated.
      3. Dedup entities within the document.
      4. Add entity nodes (idempotent on `entity.key` across docs).
      5. Emit typed edges: doc -> author, doc -> topic, doc ->
         citation, with confidence labels.
      6. Emit entity co-mention edges within the document for
         entity pairs that share the same document (INFERRED
         confidence).
    """
    extractor = extractor or EntityExtractor()
    g: nx.MultiDiGraph = nx.MultiDiGraph()

    for c in contributions:
        g.add_node(c.doc_node_id, **_document_attrs(c.metadata))

        entities = c.entities or extractor.extract(markdown=c.body_markdown, metadata=c.metadata)
        entities = deduplicate_entities(entities)

        for e in entities:
            if e.key not in g.nodes:
                g.add_node(e.key, **_entity_attrs(e))
            relation = _document_to_entity_relation(e.type)
            edge = Edge(
                source=c.doc_node_id,
                target=e.key,
                relation=relation,
                confidence=e.confidence,
                score=e.score,
                provenance=dict(e.provenance),
            )
            g.add_edge(edge.source, edge.target, **edge.as_attrs())

        # Co-mention edges between entity pairs within this document.
        # INFERRED confidence: co-mention is a heuristic signal, not
        # a direct lift from the source text.
        # All entity types are included: co-mention between authors captures
        # co-authorship signal; between citations it captures co-citation
        # signal. These are orthogonal to the typed directed edges
        # (authored_by, cites) which record provenance. Co-mention records
        # co-occurrence. Note: co_mentioned_in does not participate in the
        # doc-doc projection used by clustering (that traversal follows
        # doc→entity typed edges only), so inclusion here does not affect
        # community detection.
        keys = sorted({e.key for e in entities})
        for i, src in enumerate(keys):
            for tgt in keys[i + 1 :]:
                co = Edge(
                    source=src,
                    target=tgt,
                    relation="co_mentioned_in",
                    confidence=Confidence.INFERRED,
                    provenance={"doc_node_id": c.doc_node_id},
                )
                g.add_edge(co.source, co.target, **co.as_attrs())

    return g


def _document_to_entity_relation(entity_type: EntityType) -> str:
    if entity_type is EntityType.AUTHOR:
        return "authored_by"
    if entity_type is EntityType.CITATION:
        return "cites"
    if entity_type is EntityType.TOPIC:
        return "about"
    return f"mentions_{entity_type.value}"


# Backwards-compatibility alias for callers still using the
# paper-specific name. To be removed in a future cleanup once all
# callers (orchestrator, tests, downstream consumers) have moved
# to `DocumentContribution`.
PaperContribution = DocumentContribution
