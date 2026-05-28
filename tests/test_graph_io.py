# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Tests for `nuthatch.graph.io` (round-trip save/load)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from nuthatch.graph.io import load_graph, save_graph


def _make_graph() -> nx.MultiDiGraph:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_node("paper::p1", node_type="paper", title="A", year=2024)
    g.add_node("author::x", node_type="entity", entity_type="author", name="X")
    g.add_edge("paper::p1", "author::x", relation="authored_by", confidence="EXTRACTED")
    g.add_edge(
        "paper::p1",
        "author::x",
        relation="co_mentioned_in",
        confidence="INFERRED",
        score=0.8,
    )
    return g


class TestSaveLoad:
    def test_round_trip_preserves_nodes(self, tmp_path: Path) -> None:
        g = _make_graph()
        path = tmp_path / "g.json"
        save_graph(g, path)
        rg = load_graph(path)
        assert set(rg.nodes) == set(g.nodes)

    def test_round_trip_preserves_node_attrs(self, tmp_path: Path) -> None:
        g = _make_graph()
        path = tmp_path / "g.json"
        save_graph(g, path)
        rg = load_graph(path)
        assert rg.nodes["paper::p1"]["title"] == "A"
        assert rg.nodes["paper::p1"]["year"] == 2024

    def test_round_trip_preserves_multi_edges(self, tmp_path: Path) -> None:
        g = _make_graph()
        path = tmp_path / "g.json"
        save_graph(g, path)
        rg = load_graph(path)
        # MultiDiGraph: should have 2 edges between p1 and x.
        keys = list(rg.get_edge_data("paper::p1", "author::x").keys())
        assert len(keys) == 2

    def test_unknown_schema_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "g.json"
        path.write_text(
            '{"schema_version": 99, "graph_type": "MultiDiGraph", "data": {}}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unsupported graph schema_version"):
            load_graph(path)
