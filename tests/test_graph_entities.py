# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.graph.entities` (heuristic entity extraction)."""

from __future__ import annotations

from nuthatch.graph.edges import Confidence
from nuthatch.graph.entities import (
    EntityExtractor,
    EntityType,
    deduplicate_entities,
)


class TestAuthorsFromMetadata:
    def test_list_authors(self) -> None:
        e = EntityExtractor()
        out = e.extract(
            markdown="",
            metadata={"authors": ["Wright, S.", "Fisher, R. A."]},
        )
        names = [x.name for x in out if x.type is EntityType.AUTHOR]
        assert "Wright, S." in names and "Fisher, R. A." in names

    def test_single_author_string(self) -> None:
        e = EntityExtractor()
        out = e.extract(markdown="", metadata={"authors": "Wright, S."})
        assert any(x.type is EntityType.AUTHOR for x in out)


class TestTopicsFromMetadata:
    def test_topic_list(self) -> None:
        e = EntityExtractor()
        out = e.extract(
            markdown="",
            metadata={"topics": ["evolution", "selection"]},
        )
        topic_names = {x.name for x in out if x.type is EntityType.TOPIC}
        assert topic_names == {"evolution", "selection"}


class TestCitationsFromBody:
    def test_bracketed_citation(self) -> None:
        e = EntityExtractor()
        body = "Earlier work [Wright 1931] established the framework."
        out = e.extract(markdown=body, metadata={})
        cites = [x for x in out if x.type is EntityType.CITATION]
        assert any("Wright 1931" in c.name for c in cites)

    def test_doi_citation(self) -> None:
        e = EntityExtractor()
        body = "See doi:10.1038/nature12345 for details."
        out = e.extract(markdown=body, metadata={})
        cites = [x for x in out if x.type is EntityType.CITATION]
        assert any("10.1038/nature12345" in c.name for c in cites)

    def test_no_duplicate_citation_keys(self) -> None:
        e = EntityExtractor()
        body = "[Smith 2010] showed; later [Smith 2010] confirmed."
        out = e.extract(markdown=body, metadata={})
        cites = [x for x in out if x.type is EntityType.CITATION]
        assert len(cites) == 1


class TestDeduplication:
    def test_keeps_highest_confidence(self) -> None:
        from nuthatch.graph.entities import ExtractedEntity

        a = ExtractedEntity(
            type=EntityType.AUTHOR,
            name="X Y",
            key="author::x_y",
            confidence=Confidence.AMBIGUOUS,
        )
        b = ExtractedEntity(
            type=EntityType.AUTHOR,
            name="X Y",
            key="author::x_y",
            confidence=Confidence.EXTRACTED,
        )
        out = deduplicate_entities([a, b])
        assert len(out) == 1
        assert out[0].confidence is Confidence.EXTRACTED


class TestNerHookIsNoOpByDefault:
    def test_no_ner_in_default_extractor(self) -> None:
        e = EntityExtractor()
        # A body with prose that a NER pass would otherwise mark up.
        body = "John McDonald of Princeton studied Drosophila melanogaster."
        out = e.extract(markdown=body, metadata={})
        # No PERSON / ORG entities should appear from the default extractor.
        assert not any(x.type is EntityType.PERSON for x in out)
        assert not any(x.type is EntityType.ORG for x in out)
