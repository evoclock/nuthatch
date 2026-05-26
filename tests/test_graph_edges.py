# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.graph.edges` (Edge dataclass + confidence labels)."""

from __future__ import annotations

import pytest

from nuthatch.graph.edges import Confidence, Edge


class TestConfidence:
    def test_three_tiers_exposed(self) -> None:
        assert set(Confidence) == {
            Confidence.EXTRACTED,
            Confidence.INFERRED,
            Confidence.AMBIGUOUS,
        }


class TestEdge:
    def test_minimal_construction(self) -> None:
        e = Edge(source="a", target="b", relation="cites")
        assert e.confidence is Confidence.EXTRACTED
        assert e.score is None
        assert e.provenance == {}

    def test_as_attrs_minimal(self) -> None:
        e = Edge(source="a", target="b", relation="cites")
        attrs = e.as_attrs()
        assert attrs == {"relation": "cites", "confidence": "EXTRACTED"}

    def test_as_attrs_with_score_and_provenance(self) -> None:
        e = Edge(
            source="a",
            target="b",
            relation="co_mentioned_in",
            confidence=Confidence.INFERRED,
            score=0.72,
            provenance={"paper_node_id": "paper::001"},
        )
        attrs = e.as_attrs()
        assert attrs["relation"] == "co_mentioned_in"
        assert attrs["confidence"] == "INFERRED"
        assert attrs["score"] == 0.72
        assert attrs["provenance"] == {"paper_node_id": "paper::001"}

    def test_immutable(self) -> None:
        from dataclasses import FrozenInstanceError

        e = Edge(source="a", target="b", relation="cites")
        with pytest.raises(FrozenInstanceError):
            e.relation = "rewritten"  # type: ignore[misc]
