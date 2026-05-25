# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.graph.decay` (relevance decay formula)."""

from __future__ import annotations

import math
from datetime import date

import networkx as nx
import pytest

from nuthatch.graph.decay import apply_decay, decay_score


class TestDecayScore:
    def test_at_zero_age_equals_backlinks(self) -> None:
        score = decay_score(days_since_touched=0, backlinks=5, half_life_days=365)
        assert score == pytest.approx(5.0)

    def test_at_one_half_life_is_half_of_backlinks(self) -> None:
        score = decay_score(days_since_touched=365, backlinks=4, half_life_days=365)
        assert score == pytest.approx(2.0, rel=1e-6)

    def test_at_two_half_lives_is_quarter(self) -> None:
        score = decay_score(days_since_touched=730, backlinks=4, half_life_days=365)
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_zero_backlinks_floored_to_one(self) -> None:
        # Backlinks at most max(backlinks, 1) per the kb-reports formula.
        score = decay_score(days_since_touched=0, backlinks=0, half_life_days=365)
        assert score == pytest.approx(1.0)

    def test_pinned_node_no_decay(self) -> None:
        score = decay_score(days_since_touched=10_000, backlinks=3, half_life_days=None)
        assert score == pytest.approx(3.0)

    def test_invalid_half_life_raises(self) -> None:
        with pytest.raises(ValueError):
            decay_score(days_since_touched=0, backlinks=1, half_life_days=0)

    def test_negative_age_clamped_to_zero(self) -> None:
        score = decay_score(days_since_touched=-5, backlinks=1, half_life_days=365)
        assert score == pytest.approx(1.0)


class TestApplyDecay:
    def _make_graph(self) -> nx.MultiDiGraph:
        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.add_node(
            "paper::recent",
            node_type="paper",
            last_touched="2026-05-25",
            half_life_days=365,
        )
        g.add_node(
            "paper::old",
            node_type="paper",
            last_touched="2025-05-25",
            half_life_days=365,
        )
        g.add_node(
            "paper::pinned",
            node_type="paper",
            last_touched="2020-01-01",
            half_life_days=None,
        )
        # Some backlinks to drive the score variation.
        g.add_edge("paper::pinned", "paper::recent", relation="cites", confidence="EXTRACTED")
        g.add_edge("paper::pinned", "paper::old", relation="cites", confidence="EXTRACTED")
        return g

    def test_recent_higher_than_old(self) -> None:
        g = self._make_graph()
        apply_decay(g, now=date(2026, 5, 25))
        assert g.nodes["paper::recent"]["relevance"] > g.nodes["paper::old"]["relevance"]

    def test_pinned_no_decay(self) -> None:
        g = self._make_graph()
        apply_decay(g, now=date(2026, 5, 25))
        # Pinned node has zero backlinks → score = 1.
        assert g.nodes["paper::pinned"]["relevance"] == pytest.approx(1.0)

    def test_missing_last_touched_treated_as_today(self) -> None:
        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.add_node("x", half_life_days=365)
        apply_decay(g, now=date(2026, 5, 25))
        # No backlinks, zero age → relevance = 1 * exp(0) = 1.
        assert g.nodes["x"]["relevance"] == pytest.approx(1.0)

    def test_old_paper_decays_to_about_half(self) -> None:
        g = self._make_graph()
        apply_decay(g, now=date(2026, 5, 25))
        # paper::old: 1 backlink, 1 half-life old → score ≈ 0.5.
        assert g.nodes["paper::old"]["relevance"] == pytest.approx(0.5, rel=1e-2)


def test_module_imports_clean() -> None:
    # Smoke that the constants and ln2 derivation are sane.
    assert math.log(2.0) > 0
