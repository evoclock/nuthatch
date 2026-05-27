# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.graph.build.build_graph` (NetworkX assembly)."""

from __future__ import annotations

from nuthatch.graph.build import DocumentContribution, build_graph
from nuthatch.graph.entities import EntityExtractor, EntityType, ExtractedEntity


def _e(t: EntityType, name: str, key: str) -> ExtractedEntity:
    return ExtractedEntity(type=t, name=name, key=key)


class TestBuildGraphShape:
    def test_single_paper_adds_paper_node(self) -> None:
        contrib = DocumentContribution(
            doc_node_id="doc::p1",
            metadata={"title": "Paper One", "year": 2024},
            body_markdown="",
            entities=[],
        )
        g = build_graph([contrib])
        assert "doc::p1" in g.nodes
        assert g.nodes["doc::p1"]["node_type"] == "document"
        assert g.nodes["doc::p1"]["title"] == "Paper One"

    def test_paper_to_author_edge(self) -> None:
        contrib = DocumentContribution(
            doc_node_id="doc::p1",
            metadata={},
            body_markdown="",
            entities=[_e(EntityType.AUTHOR, "Wright", "author::wright")],
        )
        g = build_graph([contrib])
        # Edge paper -> author with relation 'authored_by'.
        edges = list(g.edges(data=True, keys=True))
        relations = {data["relation"] for _, _, _, data in edges}
        assert "authored_by" in relations

    def test_paper_to_citation_edge(self) -> None:
        contrib = DocumentContribution(
            doc_node_id="doc::p1",
            metadata={},
            body_markdown="",
            entities=[_e(EntityType.CITATION, "Smith 2010", "citation::smith_2010")],
        )
        g = build_graph([contrib])
        relations = {data["relation"] for _, _, _, data in g.edges(data=True, keys=True)}
        assert "cites" in relations


class TestEntityNodeDedup:
    def test_same_entity_two_papers_one_node(self) -> None:
        c1 = DocumentContribution(
            doc_node_id="doc::p1",
            metadata={},
            body_markdown="",
            entities=[_e(EntityType.AUTHOR, "Wright", "author::wright")],
        )
        c2 = DocumentContribution(
            doc_node_id="doc::p2",
            metadata={},
            body_markdown="",
            entities=[_e(EntityType.AUTHOR, "Wright", "author::wright")],
        )
        g = build_graph([c1, c2])
        assert g.nodes["author::wright"]["node_type"] == "entity"
        # Two paper nodes + one author node.
        paper_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "document"]
        author_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "entity"]
        assert set(paper_nodes) == {"doc::p1", "doc::p2"}
        assert author_nodes == ["author::wright"]


class TestCoMentionEdges:
    def test_co_mention_edges_within_paper(self) -> None:
        contrib = DocumentContribution(
            doc_node_id="doc::p1",
            metadata={},
            body_markdown="",
            entities=[
                _e(EntityType.AUTHOR, "Wright", "author::wright"),
                _e(EntityType.CITATION, "Smith 2010", "citation::smith_2010"),
                _e(EntityType.TOPIC, "evolution", "topic::evolution"),
            ],
        )
        g = build_graph([contrib])
        co_edges = [
            (u, v, d) for u, v, d in g.edges(data=True) if d.get("relation") == "co_mentioned_in"
        ]
        # 3 entities → C(3,2) = 3 co-mention pairs.
        assert len(co_edges) == 3
        for _, _, d in co_edges:
            assert d["confidence"] == "INFERRED"


class TestEntityExtractorFallback:
    def test_runs_extractor_when_entities_empty(self) -> None:
        contrib = DocumentContribution(
            doc_node_id="doc::p1",
            metadata={"authors": ["Wright"]},
            body_markdown="See [Smith 2010] for context.",
            entities=[],
        )
        g = build_graph([contrib], extractor=EntityExtractor())
        # Paper plus 2 entities (Wright author, Smith 2010 citation).
        entity_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "entity"]
        assert set(entity_nodes) == {"author::wright", "citation::smith_2010"}
