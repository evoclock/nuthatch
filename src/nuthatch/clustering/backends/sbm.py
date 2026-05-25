# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Stochastic Block Model clustering via Tiago Peixoto's `graph-tool`.

This is the principled / Bayesian backend (`Rigor.PRINCIPLED`).
Uses the nested degree-corrected SBM from `graph-tool`, which
fits a hierarchical block model and returns the maximum-a-
posteriori partition without needing a resolution parameter.

`graph-tool` is conda-only (no PyPI wheels). nuthatch ships it via
a dedicated conda env (`nuthatch-gt`). When the active Python
interpreter does not have `graph_tool` importable, `available()`
returns False and the router falls back to Leiden / embeddings.

Inputs at `cluster`: a `ClusteringRequest` carrying a JSON-encoded
    networkx graph snapshot (via `node_link_data`).

Outputs: a `ClusteringResponse` with the partition (node -> block),
    rigor (`PRINCIPLED`), and the serialised `BlockState` for
    downstream consumers that want to inspect the model.

Pattern: standard `graph_tool.inference.minimize_nested_blockmodel_dl`
usage. No reuse from kestrel (which uses Leiden / Louvain, not
SBM); this is a fresh implementation against the graph-tool API.

Assumptions: graph is undirected for the SBM fit (directed edges
    are accepted in the snapshot but block-modelled as undirected).
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


def _import_graph_tool() -> Any:
    try:
        import graph_tool.all as gt  # noqa: PLC0415

        return gt
    except ImportError:
        return None


class SBMBackend:
    """`ClusteringBackend`-compliant Bayesian SBM partitioner."""

    name: str = "sbm_graph_tool"
    rigor: Rigor = Rigor.PRINCIPLED
    location: BackendLocation = BackendLocation.LOCAL

    def __init__(
        self,
        *,
        degree_corrected: bool = True,
        nested: bool = True,
    ) -> None:
        self.degree_corrected = degree_corrected
        self.nested = nested

    def available(self) -> bool:
        return _import_graph_tool() is not None

    def cluster(self, request: ClusteringRequest) -> ClusteringResponse:
        gt = _import_graph_tool()
        if gt is None:
            raise RuntimeError(
                "graph_tool not available; activate the `nuthatch-gt` "
                "conda env or install via conda-forge."
            )

        payload = json.loads(request.graph_snapshot.decode("utf-8"))
        nxg = nx.node_link_graph(payload, edges="edges")
        gtg, node_id_by_index = _nx_to_graph_tool(nxg, gt)

        t0 = time.perf_counter()
        if self.nested:
            state = gt.minimize_nested_blockmodel_dl(
                gtg,
                state_args={"deg_corr": self.degree_corrected},
            )
            partition_array = state.get_bs()[0]  # top-level partition
        else:
            state = gt.minimize_blockmodel_dl(
                gtg,
                state_args={"deg_corr": self.degree_corrected},
            )
            partition_array = state.get_blocks().a
        elapsed = time.perf_counter() - t0

        partition = {
            node_id_by_index[i]: int(b)
            for i, b in enumerate(partition_array)
        }

        return ClusteringResponse(
            partition=partition,
            rigor_used=self.rigor,
            runtime_seconds=elapsed,
            backend_used=self.name,
            block_state=None,  # in-memory only; pickling graph-tool state is fragile
            notes=(
                f"nested={self.nested}, "
                f"degree_corrected={self.degree_corrected}"
            ),
        )


def _nx_to_graph_tool(
    nx_graph: nx.Graph,
    gt: Any,
) -> tuple[Any, dict[int, str]]:
    """Convert a networkx graph to a graph-tool graph.

    Returns `(gt_graph, index -> nx_node_id)` so the caller can map
    block-array positions back to the original node IDs.
    """
    g = gt.Graph(directed=False)
    name_to_index: dict[str, int] = {}
    index_to_name: dict[int, str] = {}
    for node in nx_graph.nodes():
        v = g.add_vertex()
        idx = int(v)
        name_to_index[str(node)] = idx
        index_to_name[idx] = str(node)
    for u, v in nx_graph.edges():
        g.add_edge(name_to_index[str(u)], name_to_index[str(v)])
    return g, index_to_name
