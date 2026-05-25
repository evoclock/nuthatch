# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Clustering router: pick the highest-rigor available backend.

Purpose: nuthatch ships three backends (SBM, Leiden, embeddings)
    behind the `ClusteringBackend` protocol. At runtime not all
    are necessarily available (graph-tool is conda-only,
    leidenalg may not be installed, the embeddings backend needs
    pre-computed vectors). The router picks the highest-rigor
    backend that can actually run for the given request and
    surfaces the choice + any downgrade in the response.

Inputs: a list of candidate backends + the request.

Outputs: the response from whichever backend ran, with `rigor_used`
    reflecting the actual tier (not the requested tier when a
    downgrade occurred) and `notes` recording the downgrade
    rationale.

Per DECISIONS.md, downgrades are surfaced honestly to the user
(via `rigor_used`); never silently substituted.
"""

from __future__ import annotations

from collections.abc import Sequence

from nuthatch.clustering.backends.embeddings import EmbeddingsBackend
from nuthatch.clustering.backends.leiden import LeidenBackend
from nuthatch.clustering.backends.sbm import SBMBackend
from nuthatch.clustering.protocol import (
    ClusteringBackend,
    ClusteringRequest,
    ClusteringResponse,
    Rigor,
)

# Rigor ordering (highest first). The router walks this order to
# pick the most-rigorous available backend that satisfies the
# request.
_RIGOR_ORDER: tuple[Rigor, ...] = (
    Rigor.PRINCIPLED,
    Rigor.HEURISTIC,
    Rigor.EMBEDDINGS_ONLY,
)


def default_backends() -> list[ClusteringBackend]:
    """Return the three standard backends, instantiated."""
    return [SBMBackend(), LeidenBackend(), EmbeddingsBackend()]


class ClusteringRouter:
    """Routes a `ClusteringRequest` to the most-rigorous available backend.

    Honours the request's `rigor` preference as a CEILING: if the
    request asks for `HEURISTIC` the router will not promote to
    `PRINCIPLED`, only downgrade if needed.
    """

    __slots__ = ("_backends",)

    def __init__(
        self,
        backends: Sequence[ClusteringBackend] | None = None,
    ) -> None:
        self._backends = list(backends) if backends is not None else default_backends()

    @property
    def backends(self) -> list[ClusteringBackend]:
        return list(self._backends)

    def pick(self, request: ClusteringRequest) -> ClusteringBackend | None:
        """Return the chosen backend or `None` if nothing is available."""
        ceiling_index = _RIGOR_ORDER.index(request.rigor)
        # Start at the requested ceiling, walk down rigor order.
        for tier in _RIGOR_ORDER[ceiling_index:]:
            for backend in self._backends:
                if backend.rigor is tier and backend.available():
                    return backend
        return None

    def cluster(self, request: ClusteringRequest) -> ClusteringResponse:
        backend = self.pick(request)
        if backend is None:
            raise RuntimeError(
                "no clustering backend available; install at least "
                "one of: graph-tool (conda), leidenalg + python-igraph, "
                "or scikit-learn (transitively pulled by sentence-transformers)"
            )
        response = backend.cluster(request)
        # Annotate downgrade if the chosen rigor is below the request.
        if response.rigor_used is not request.rigor:
            requested_index = _RIGOR_ORDER.index(request.rigor)
            used_index = _RIGOR_ORDER.index(response.rigor_used)
            if used_index > requested_index:
                # Pure dataclass; build a fresh response to amend notes.
                response = ClusteringResponse(
                    partition=response.partition,
                    rigor_used=response.rigor_used,
                    runtime_seconds=response.runtime_seconds,
                    backend_used=response.backend_used,
                    block_state=response.block_state,
                    notes=(
                        f"{response.notes} | downgraded from "
                        f"{request.rigor.value} (requested) to "
                        f"{response.rigor_used.value} (only available)"
                    ),
                )
        return response
