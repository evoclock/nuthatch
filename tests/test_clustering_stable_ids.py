# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.clustering.stable_ids.remap_to_previous`."""

from __future__ import annotations

from nuthatch.clustering.stable_ids import remap_to_previous


class TestRemapToPrevious:
    def test_empty_new_returns_empty(self) -> None:
        remapped, id_map = remap_to_previous({}, {"a": 0, "b": 1})
        assert remapped == {}
        assert id_map == {}

    def test_perfect_overlap_preserves_ids(self) -> None:
        previous = {"a": 0, "b": 0, "c": 1, "d": 1}
        new = {"a": 5, "b": 5, "c": 7, "d": 7}
        remapped, id_map = remap_to_previous(new, previous)
        assert remapped == previous
        assert id_map[5] == 0
        assert id_map[7] == 1

    def test_partial_overlap_inherits_largest_intersection(self) -> None:
        previous = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1}
        # New community {a,b,c,d} has overlap 3 with old 0 and 1 with old 1.
        new = {"a": 9, "b": 9, "c": 9, "d": 9, "e": 9}
        remapped, _ = remap_to_previous(new, previous)
        assert remapped["a"] == 0  # inherits old 0

    def test_no_overlap_gets_fresh_id_avoiding_reserved(self) -> None:
        previous = {"a": 0, "b": 1, "c": 2}
        new = {"x": 7, "y": 7}
        remapped, id_map = remap_to_previous(new, previous)
        # Fresh ID should not collide with 0, 1, 2.
        assert id_map[7] not in {0, 1, 2}

    def test_greedy_first_come_for_size(self) -> None:
        # Larger new community should claim the matching old ID first.
        previous = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 1, "f": 1}
        new = {"a": 5, "b": 5, "c": 5, "d": 5, "e": 6, "f": 6}
        # New community 5 (4 members, overlaps old 0) should map to 0;
        # new community 6 (2 members, overlaps old 1) should map to 1.
        remapped, id_map = remap_to_previous(new, previous)
        assert id_map[5] == 0
        assert id_map[6] == 1
