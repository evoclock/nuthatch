# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for `nuthatch.clustering.hub_exclusion`."""

from __future__ import annotations

import networkx as nx
import pytest

from nuthatch.clustering.hub_exclusion import (
    core_nodes,
    exclude_core_nodes,
    reattach_by_majority_neighbour,
)


def _star_graph(n_leaves: int = 10) -> nx.Graph:
    g = nx.Graph()
    g.add_node("hub")
    for i in range(n_leaves):
        leaf = f"leaf{i}"
        g.add_node(leaf)
        g.add_edge("hub", leaf)
    return g


class TestCoreNodes:
    def test_empty_graph_returns_empty(self) -> None:
        assert core_nodes(nx.Graph()) == []

    def test_star_graph_picks_hub(self) -> None:
        g = _star_graph(20)
        cores = core_nodes(g, percentile=99.0)
        assert "hub" in cores

    def test_low_percentile_picks_more(self) -> None:
        g = _star_graph(20)
        many = core_nodes(g, percentile=50.0)
        few = core_nodes(g, percentile=99.0)
        assert len(many) >= len(few)

    def test_percentile_validation(self) -> None:
        with pytest.raises(ValueError):
            core_nodes(_star_graph(5), percentile=-1.0)
        with pytest.raises(ValueError):
            core_nodes(_star_graph(5), percentile=101.0)


class TestExcludeCoreNodes:
    def test_excludes_hub_returns_leaves_only(self) -> None:
        g = _star_graph(10)
        sub, excluded = exclude_core_nodes(g, percentile=99.0)
        assert "hub" in excluded
        assert "hub" not in sub.nodes()
        assert all(n.startswith("leaf") for n in sub.nodes())

    def test_subgraph_is_copy_not_view(self) -> None:
        g = _star_graph(5)
        sub, _ = exclude_core_nodes(g, percentile=99.0)
        sub.add_node("test_mutation")
        assert "test_mutation" not in g.nodes()


class TestReattachByMajority:
    def test_reattach_picks_majority_community(self) -> None:
        # Hub with 3 leaves in community 0 and 2 leaves in community 1.
        g = nx.Graph()
        for n in ["hub", "a", "b", "c", "d", "e"]:
            g.add_node(n)
        for leaf in ["a", "b", "c", "d", "e"]:
            g.add_edge("hub", leaf)
        partition = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1}
        merged = reattach_by_majority_neighbour(g, partition, ["hub"])
        assert merged["hub"] == 0

    def test_isolated_node_gets_fresh_community(self) -> None:
        g = nx.Graph()
        g.add_node("alone")
        merged = reattach_by_majority_neighbour(g, {}, ["alone"])
        assert "alone" in merged

    def test_tie_breaks_to_lower_id(self) -> None:
        g = nx.Graph()
        for n in ["hub", "a", "b"]:
            g.add_node(n)
        g.add_edge("hub", "a")
        g.add_edge("hub", "b")
        merged = reattach_by_majority_neighbour(g, {"a": 5, "b": 3}, ["hub"])
        assert merged["hub"] == 3
