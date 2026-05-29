# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Graph (de)serialisation. Round-trippable to and from `.kg/graph/graph.json`.

Purpose: persist the corpus graph so it survives across `nuthatch
    ingest` runs, and reload it for the next ingest pass without
    re-extracting. JSON because the graph is small (10^4 - 10^6
    edges for a typical paper corpus) and human-inspectable; the
    node/edge attribute payloads carry enough provenance that a
    round-trip preserves semantics.

Inputs at save: an `nx.MultiDiGraph` + a target path.

Outputs at load: an `nx.MultiDiGraph` rebuilt from the JSON.

Format: networkx `node_link_data` with `edges=True` (per the modern
networkx API; the older default of `links=True` is deprecated).
Wrapped in a small envelope with a schema version so a future
nuthatch can detect old graph files and migrate.

Assumptions: edge attributes are JSON-serialisable. Provenance
    dicts and dataclass values are stringified at the build layer
    before they enter the graph; this module does not transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

_SCHEMA_VERSION: int = 1


def save_graph(g: nx.MultiDiGraph, path: Path) -> None:
    """Write `g` to `path` as JSON with a schema-version envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "graph_type": "MultiDiGraph",
        "data": nx.node_link_data(g, edges="edges"),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Read `path` back into an `nx.MultiDiGraph`.

    Raises `ValueError` on unknown schema versions so callers can
    detect old graph files and migrate.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported graph schema_version: {schema_version} (expected {_SCHEMA_VERSION})"
        )
    graph_type = payload.get("graph_type", "MultiDiGraph")
    if graph_type != "MultiDiGraph":
        raise ValueError(f"unsupported graph_type: {graph_type}")
    g: nx.MultiDiGraph = nx.node_link_graph(
        payload["data"], directed=True, multigraph=True, edges="edges"
    )
    return g
