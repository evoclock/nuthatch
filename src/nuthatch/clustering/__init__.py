# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Clustering: protocol + backends + router + hub exclusion + stable IDs.

The `ClusteringBackend` protocol from Sprint 0 is re-exported here
alongside the concrete backends (Sprint 5) and the supporting
machinery (hub exclusion via `core_nodes`, community-ID remap).
"""

from nuthatch.clustering.hub_exclusion import core_nodes, exclude_core_nodes
from nuthatch.clustering.protocol import (
    BackendLocation,
    ClusteringBackend,
    ClusteringRequest,
    ClusteringResponse,
    Rigor,
)
from nuthatch.clustering.router import ClusteringRouter
from nuthatch.clustering.stable_ids import remap_to_previous

__all__ = [
    "BackendLocation",
    "ClusteringBackend",
    "ClusteringRequest",
    "ClusteringResponse",
    "ClusteringRouter",
    "Rigor",
    "core_nodes",
    "exclude_core_nodes",
    "remap_to_previous",
]
