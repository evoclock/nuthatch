# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Leiden clustering backend (`leidenalg` + `python-igraph`).

Uses the canonical Leiden implementation by Vincent Traag
(co-author of the Leiden algorithm), via the `leidenalg` package
operating on a `python-igraph` graph. This is the heuristic /
medium-rigor backend; the principled-rigor backend is SBM
(`backends/sbm.py`) and the bridge backend is embeddings
(`backends/embeddings.py`).

Inputs at `cluster`: a `ClusteringRequest` carrying a serialised
    `nx.Graph` snapshot (JSON via networkx `node_link_data`).

Outputs: a `ClusteringResponse` with the partition (node -> community
    id), the rigor tier (`HEURISTIC`), and timing.

Assumptions: `leidenalg` and `python-igraph` are available. If
    not, `available()` returns False and the router falls back.
"""

from __future__ import annotations

import json
import time
from typing import Any

import networkx as nx

from nuthatch.clustering.protocol import (
    BackendLocation,
    ClusteringRequest,
    ClusteringResponse,
    Rigor,
)


def _import_or_none() -> tuple[Any, Any]:
    try:
        import igraph
        import leidenalg

        return leidenalg, igraph
    except ImportError:
        return None, None


class LeidenBackend:
    """`ClusteringBackend`-compliant Leiden partitioner."""

    name: str = "leiden"
    rigor: Rigor = Rigor.HEURISTIC
    location: BackendLocation = BackendLocation.LOCAL

    def __init__(self, *, resolution: float = 1.0, seed: int = 42) -> None:
        self.resolution = resolution
        self.seed = seed

    def available(self) -> bool:
        leidenalg, igraph = _import_or_none()
        return leidenalg is not None and igraph is not None

    def cluster(self, request: ClusteringRequest) -> ClusteringResponse:
        leidenalg, igraph = _import_or_none()
        if leidenalg is None or igraph is None:
            raise RuntimeError(
                "leidenalg + python-igraph not available; "
                "install with `sfw uv add leidenalg python-igraph`"
            )

        payload = json.loads(request.graph_snapshot.decode("utf-8"))
        nxg = nx.node_link_graph(payload, edges="edges")
        ig = _nx_to_igraph(nxg, igraph)

        t0 = time.perf_counter()
        partition = leidenalg.find_partition(
            ig,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            seed=self.seed,
        )
        elapsed = time.perf_counter() - t0

        membership = partition.membership
        node_ids = [v["name"] for v in ig.vs]
        partition_dict = dict(zip(node_ids, membership, strict=True))

        return ClusteringResponse(
            partition=partition_dict,
            rigor_used=self.rigor,
            runtime_seconds=elapsed,
            backend_used=self.name,
            notes=f"leiden via leidenalg; resolution={self.resolution}",
        )


def _nx_to_igraph(nx_graph: Any, igraph_module: Any) -> Any:
    """Convert a networkx graph to igraph, preserving node IDs in `name`."""
    nodes = list(nx_graph.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    edges = [(node_index[u], node_index[v]) for u, v in nx_graph.edges()]
    ig = igraph_module.Graph(n=len(nodes), edges=edges, directed=False)
    ig.vs["name"] = [str(n) for n in nodes]
    return ig
