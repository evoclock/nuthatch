# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Clustering backend protocol; the first concrete spec artifact.

Defines the abstraction that lets nuthatch route a clustering request
to any of three backend kinds at runtime:

- **principled**: Bayesian Stochastic Block Model via
  Tiago Peixoto's `graph-tool`. The intended default when graph-tool is
  installed and the graph is small enough to fit local compute.
- **heuristic**: Leiden / Louvain via `networkx` or `graspologic`.
  Faster, but inherits the modularity resolution-limit + false-
  positive failure modes documented in `docs/DECISIONS.md`.
- **embeddings_only**: no graph clustering at all; falls back to
  vector similarity over a sentence-transformer embedding store
  (`chromadb`). Used during the gap between a graph mutation and
  the next user-triggered SBM refit, so the system stays usable
  without a stale graph partition.

The same protocol shape applies whether the compute runs locally or
on a remote service. The OSS path always supports `local`; remote
backends (hosted SaaS, managed-on-customer-cloud) ship as separate
packages that register additional `ClusteringBackend` implementations.

No implementation lives here yet: only the protocol. Implementations
land as separate modules once we have an ingested graph to cluster.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable


class Rigor(StrEnum):
    """How rigorous the clustering result is.

    Surfaced in every `ClusteringResponse` so downstream code
    (LLM-context extractor, dashboard, MCP server) can honestly
    label the partition it's working with.
    """

    PRINCIPLED = "principled"  # SBM-grade (Bayesian, no resolution param)
    HEURISTIC = "heuristic"  # Leiden / Louvain (modularity-based)
    EMBEDDINGS_ONLY = "embeddings_only"  # no graph partition; vector-similarity bridge


class BackendLocation(StrEnum):
    """Where the clustering compute physically runs."""

    LOCAL = "local"  # this machine
    REMOTE_SAAS = "remote-saas"  # nuthatch hosted SaaS
    REMOTE_CUSTOMER_CLOUD = "remote-customer-cloud"  # managed-on-customer-cloud (Model C)


class ClusteringRequest:
    """Payload sent to a clustering backend.

    Carries the graph snapshot (serialised; backend-agnostic format
    TBD: likely GraphML or a custom compact form) plus the
    requested rigor and any backend-specific hints.

    Concrete shape will harden once the graph schema is settled; for
    now this is a stub that pins the API surface.
    """

    __slots__ = ("backend_hint", "graph_snapshot", "rigor")

    def __init__(
        self,
        graph_snapshot: bytes,
        *,
        rigor: Rigor = Rigor.PRINCIPLED,
        backend_hint: BackendLocation | None = None,
    ) -> None:
        self.graph_snapshot = graph_snapshot
        self.rigor = rigor
        self.backend_hint = backend_hint


class ClusteringResponse:
    """What a clustering backend returns.

    The `rigor_used` field MAY differ from the requested rigor (e.g.,
    a principled request downgraded to heuristic when graph-tool is
    unavailable). The downstream consumer is expected to surface this
    to the user honestly, not silently accept the downgrade.
    """

    __slots__ = (
        "backend_used",
        "block_state",
        "notes",
        "partition",
        "rigor_used",
        "runtime_seconds",
    )

    def __init__(
        self,
        *,
        partition: dict[str, int],
        rigor_used: Rigor,
        runtime_seconds: float,
        backend_used: str,
        block_state: bytes | None = None,
        notes: str = "",
    ) -> None:
        self.partition = partition
        self.rigor_used = rigor_used
        self.runtime_seconds = runtime_seconds
        self.backend_used = backend_used
        # Serialised SBM BlockState (for principled-rigor responses);
        # None for heuristic / embeddings-only backends.
        self.block_state = block_state
        # Caveats / warnings to surface to the user (e.g. "downgraded
        # to heuristic because graph-tool not installed").
        self.notes = notes


@runtime_checkable
class ClusteringBackend(Protocol):
    """Protocol every clustering backend implements.

    Backends are registered with a plugin entry-point or imported
    explicitly; the routing layer picks the highest-rigor available
    backend that matches the request, with explicit user-visible
    fallback when a downgrade happens.
    """

    @property
    def name(self) -> str:
        """Short identifier for telemetry / error messages."""
        ...

    @property
    def rigor(self) -> Rigor:
        """The rigor tier this backend provides if it can run at all."""
        ...

    @property
    def location(self) -> BackendLocation:
        """Where the compute runs."""
        ...

    def available(self) -> bool:
        """True if this backend can serve a request right now.

        Implementations check for required runtime (graph-tool, a
        remote endpoint's health, etc.) and return False rather than
        raising, so the router can pick a fallback cleanly.
        """
        ...

    def cluster(self, request: ClusteringRequest) -> ClusteringResponse:
        """Run the clustering and return the partition + metadata."""
        ...
