# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for the concrete clustering backends + the router.

Focus: availability + dispatch logic. Real-cluster integration
tests (running SBM on a real graph) live in a separate slow-test
module; this file uses synthetic data and fake backends.
"""

from __future__ import annotations

import json

import networkx as nx
import pytest

from nuthatch.clustering.backends.embeddings import EmbeddingsBackend
from nuthatch.clustering.backends.leiden import LeidenBackend
from nuthatch.clustering.backends.sbm import SBMBackend
from nuthatch.clustering.protocol import (
    BackendLocation,
    ClusteringRequest,
    ClusteringResponse,
    Rigor,
)
from nuthatch.clustering.router import ClusteringRouter


def _nx_snapshot() -> bytes:
    g: nx.Graph = nx.karate_club_graph()
    payload = nx.node_link_data(g, edges="edges")
    return json.dumps(payload, default=str).encode("utf-8")


def _embeddings_snapshot(n: int = 8, dim: int = 4) -> bytes:
    return json.dumps(
        {
            "node_ids": [f"chunk{i}" for i in range(n)],
            "embeddings": [[float(i + d) for d in range(dim)] for i in range(n)],
        }
    ).encode("utf-8")


class TestEmbeddingsBackend:
    def test_available_with_sklearn(self) -> None:
        # sklearn is pulled transitively; expect True in this venv.
        assert EmbeddingsBackend().available()

    def test_partition_shape(self) -> None:
        backend = EmbeddingsBackend(n_clusters=3)
        req = ClusteringRequest(
            graph_snapshot=_embeddings_snapshot(),
            rigor=Rigor.EMBEDDINGS_ONLY,
        )
        rv = backend.cluster(req)
        assert rv.rigor_used is Rigor.EMBEDDINGS_ONLY
        assert rv.backend_used == "embeddings_kmeans"
        assert len(rv.partition) == 8
        # 3 clusters or fewer (k may be reduced if n_points smaller).
        assert max(rv.partition.values()) < 3


class TestLeidenBackend:
    def test_available(self) -> None:
        # leidenalg + igraph were installed for this sprint.
        assert LeidenBackend().available()

    def test_clusters_karate(self) -> None:
        backend = LeidenBackend()
        req = ClusteringRequest(graph_snapshot=_nx_snapshot(), rigor=Rigor.HEURISTIC)
        rv = backend.cluster(req)
        assert rv.rigor_used is Rigor.HEURISTIC
        assert rv.backend_used == "leiden"
        # Karate club has 34 nodes.
        assert len(rv.partition) == 34
        # Expect at least 2 communities, no more than 10 on a 34-node graph.
        n_communities = len(set(rv.partition.values()))
        assert 2 <= n_communities <= 10


class TestSBMBackendAvailability:
    def test_available_reflects_graph_tool_import(self) -> None:
        # graph_tool is in the conda env, not the uv venv; expect False
        # unless the test runner happens to have access. Don't assert
        # the value, just exercise the path.
        SBMBackend().available()


class _FakeBackend:
    def __init__(self, name: str, rigor: Rigor, available: bool) -> None:
        self.name = name
        self.rigor = rigor
        self.location = BackendLocation.LOCAL
        self._available = available

    def available(self) -> bool:
        return self._available

    def cluster(self, request: ClusteringRequest) -> ClusteringResponse:
        return ClusteringResponse(
            partition={"a": 0},
            rigor_used=self.rigor,
            runtime_seconds=0.0,
            backend_used=self.name,
        )


class TestClusteringRouter:
    def test_picks_highest_available(self) -> None:
        router = ClusteringRouter(
            backends=[
                _FakeBackend("sbm", Rigor.PRINCIPLED, available=False),
                _FakeBackend("leiden", Rigor.HEURISTIC, available=True),
                _FakeBackend("emb", Rigor.EMBEDDINGS_ONLY, available=True),
            ]
        )
        req = ClusteringRequest(graph_snapshot=b"{}", rigor=Rigor.PRINCIPLED)
        rv = router.cluster(req)
        assert rv.backend_used == "leiden"
        assert "downgraded" in (rv.notes or "")

    def test_picks_principled_when_available(self) -> None:
        router = ClusteringRouter(
            backends=[
                _FakeBackend("sbm", Rigor.PRINCIPLED, available=True),
                _FakeBackend("leiden", Rigor.HEURISTIC, available=True),
            ]
        )
        req = ClusteringRequest(graph_snapshot=b"{}", rigor=Rigor.PRINCIPLED)
        rv = router.cluster(req)
        assert rv.backend_used == "sbm"
        # No downgrade note because we got the requested rigor.
        assert "downgraded" not in (rv.notes or "")

    def test_respects_ceiling(self) -> None:
        # User requested HEURISTIC; router should NOT promote to PRINCIPLED.
        router = ClusteringRouter(
            backends=[
                _FakeBackend("sbm", Rigor.PRINCIPLED, available=True),
                _FakeBackend("leiden", Rigor.HEURISTIC, available=True),
            ]
        )
        req = ClusteringRequest(graph_snapshot=b"{}", rigor=Rigor.HEURISTIC)
        rv = router.cluster(req)
        assert rv.backend_used == "leiden"

    def test_no_backend_raises(self) -> None:
        router = ClusteringRouter(
            backends=[
                _FakeBackend("emb", Rigor.EMBEDDINGS_ONLY, available=False),
            ]
        )
        req = ClusteringRequest(graph_snapshot=b"{}", rigor=Rigor.PRINCIPLED)
        with pytest.raises(RuntimeError, match="no clustering backend"):
            router.cluster(req)
