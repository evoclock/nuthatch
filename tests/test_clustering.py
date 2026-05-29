# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the `ClusteringBackend` protocol shape.

The protocol itself has no behaviour yet; these tests pin the API
surface so a future implementation that diverges from it is caught
at PR time rather than at integration time.
"""

from __future__ import annotations

from nuthatch.clustering import (
    BackendLocation,
    ClusteringBackend,
    ClusteringRequest,
    ClusteringResponse,
    Rigor,
)


class TestRigorEnum:
    def test_three_tiers_exist(self) -> None:
        # The user-facing rigor commitment: every clustering result
        # MUST be labelled with one of these three. Pinning the set
        # so an accidental addition / removal is noticed.
        assert {member.value for member in Rigor} == {
            "principled",
            "heuristic",
            "embeddings_only",
        }


class TestBackendLocationEnum:
    def test_three_locations_exist(self) -> None:
        # Mirrors the OSS / paid split: local + the two paid backends.
        assert {member.value for member in BackendLocation} == {
            "local",
            "remote-saas",
            "remote-customer-cloud",
        }


class TestClusteringRequest:
    def test_constructs_with_defaults(self) -> None:
        req = ClusteringRequest(graph_snapshot=b"")
        assert req.rigor == Rigor.PRINCIPLED
        assert req.backend_hint is None

    def test_constructs_with_explicit_rigor(self) -> None:
        req = ClusteringRequest(
            graph_snapshot=b"",
            rigor=Rigor.HEURISTIC,
            backend_hint=BackendLocation.LOCAL,
        )
        assert req.rigor == Rigor.HEURISTIC
        assert req.backend_hint == BackendLocation.LOCAL


class TestClusteringResponse:
    def test_constructs_with_required_fields(self) -> None:
        resp = ClusteringResponse(
            partition={"a": 0, "b": 0, "c": 1},
            rigor_used=Rigor.PRINCIPLED,
            runtime_seconds=1.5,
            backend_used="graph-tool-sbm",
        )
        assert resp.partition == {"a": 0, "b": 0, "c": 1}
        assert resp.rigor_used == Rigor.PRINCIPLED
        assert resp.block_state is None
        assert resp.notes == ""


class TestProtocolShape:
    def test_protocol_is_runtime_checkable(self) -> None:
        # `isinstance(obj, ClusteringBackend)` must work so the router
        # can validate registered backends at startup.

        class _Stub:
            name = "stub"
            rigor = Rigor.EMBEDDINGS_ONLY
            location = BackendLocation.LOCAL

            def available(self) -> bool:
                return False

            def cluster(self, request: ClusteringRequest) -> ClusteringResponse:
                raise NotImplementedError

        assert isinstance(_Stub(), ClusteringBackend)
