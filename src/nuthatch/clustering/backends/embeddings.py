# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Embedding-based clustering backend (k-means on chunk vectors).

This is the bridge / lowest-rigor backend (`EMBEDDINGS_ONLY`).
Used when neither SBM nor Leiden is available, or when the user
explicitly wants to cluster purely by semantic similarity rather
than graph structure (e.g. corpus with sparse / no citation
structure).

Inputs at `cluster`: a `ClusteringRequest` whose `graph_snapshot`
    is a JSON blob with two arrays: `node_ids` (list[str]) and
    `embeddings` (list[list[float]]). This is a deliberate
    deviation from the graph-shape snapshot the other backends
    consume — this backend doesn't use graph structure at all.

Outputs: a `ClusteringResponse` with the partition (node -> cluster
    index), rigor (`EMBEDDINGS_ONLY`), and timing.

Assumptions: scikit-learn is available (transitively pulled by
    sentence-transformers). The number of clusters is configurable;
    default uses the elbow / silhouette heuristic up to a cap.
"""

from __future__ import annotations

import json
import time
from typing import Any

from nuthatch.clustering.protocol import (
    BackendLocation,
    ClusteringRequest,
    ClusteringResponse,
    Rigor,
)


def _import_sklearn() -> Any:
    try:
        from sklearn.cluster import KMeans

        return KMeans
    except ImportError:
        return None


class EmbeddingsBackend:
    """`ClusteringBackend`-compliant k-means partitioner over embeddings."""

    name: str = "embeddings_kmeans"
    rigor: Rigor = Rigor.EMBEDDINGS_ONLY
    location: BackendLocation = BackendLocation.LOCAL

    def __init__(
        self,
        *,
        n_clusters: int | None = None,
        max_clusters: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

    def available(self) -> bool:
        return _import_sklearn() is not None

    def cluster(self, request: ClusteringRequest) -> ClusteringResponse:
        KMeans = _import_sklearn()
        if KMeans is None:
            raise RuntimeError(
                "scikit-learn not available; should be pulled by sentence-transformers"
            )

        payload = json.loads(request.graph_snapshot.decode("utf-8"))
        node_ids: list[str] = payload["node_ids"]
        embeddings: list[list[float]] = payload["embeddings"]
        if len(node_ids) != len(embeddings):
            raise ValueError("node_ids and embeddings length mismatch")
        if not node_ids:
            return ClusteringResponse(
                partition={},
                rigor_used=self.rigor,
                runtime_seconds=0.0,
                backend_used=self.name,
                notes="empty input",
            )

        k = self._pick_k(len(node_ids))

        t0 = time.perf_counter()
        model = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
        labels = model.fit_predict(embeddings)
        elapsed = time.perf_counter() - t0

        partition = {nid: int(lbl) for nid, lbl in zip(node_ids, labels, strict=True)}
        return ClusteringResponse(
            partition=partition,
            rigor_used=self.rigor,
            runtime_seconds=elapsed,
            backend_used=self.name,
            notes=f"k={k}",
        )

    def _pick_k(self, n_points: int) -> int:
        if self.n_clusters is not None:
            return max(1, min(self.n_clusters, n_points))
        # Heuristic: sqrt(n / 2), capped at `max_clusters`, floored at 2.
        import math

        k = max(2, min(self.max_clusters, int(math.sqrt(max(n_points / 2, 1)))))
        return min(k, n_points)
