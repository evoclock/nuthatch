# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Community-ID stability across successive partitions.

Purpose: when a corpus is re-clustered (new papers added, refit
    triggered), the partitioner returns community IDs that don't
    necessarily match the previous run. For UX continuity (saved
    queries, citation patterns, Obsidian wikilinks pointing at
    `community::42`) we want yesterday's "community 42" to remain
    "community 42" today if the membership overlaps enough.

Inputs at `remap_to_previous`: the new partition (node -> new_id)
    and the previous partition (node -> old_id).

Outputs: a remapped new partition (node -> old_id-where-possible)
    plus a mapping of new_id -> remapped_id.

Algorithm: greedy set-overlap. For each new community (largest
    first), find the previous community with the largest
    intersection. If the intersection is non-empty, the new
    community inherits that previous ID. Tie-break by lower
    previous ID. Communities with no overlap get fresh IDs that
    don't collide with any old ID.

No equivalent function exists in the currently-installed a prior implementation
version (the SPRINT_PLAN.md reuse map referenced one but it has
since been removed upstream); this is a fresh implementation
shaped by the standard greedy-overlap pattern used in the
community-detection literature.
"""

from __future__ import annotations

from collections.abc import Iterable


def _invert(partition: dict[str, int]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for node, community in partition.items():
        out.setdefault(community, set()).add(node)
    return out


def remap_to_previous(
    new_partition: dict[str, int],
    previous_partition: dict[str, int],
) -> tuple[dict[str, int], dict[int, int]]:
    """Remap `new_partition` IDs to match `previous_partition` where overlap allows.

    Returns:
        remapped_partition: same node IDs, communities renumbered to
            inherit previous IDs where overlap permits.
        id_map: new_id -> chosen_id (the chosen id is the previous
            community's ID when overlap won, otherwise a fresh
            unused integer).
    """
    if not new_partition:
        return {}, {}

    new_buckets = _invert(new_partition)
    old_buckets = _invert(previous_partition)

    # Sort new communities by size (largest first) for greedy match.
    new_communities_by_size = sorted(new_buckets.items(), key=lambda kv: -len(kv[1]))

    used_old_ids: set[int] = set()
    id_map: dict[int, int] = {}

    for new_id, new_members in new_communities_by_size:
        # Find the previous community with the largest unclaimed overlap.
        best_overlap = 0
        best_old_id: int | None = None
        for old_id, old_members in sorted(old_buckets.items()):
            if old_id in used_old_ids:
                continue
            overlap = len(new_members & old_members)
            if overlap > best_overlap:
                best_overlap = overlap
                best_old_id = old_id
        if best_old_id is not None and best_overlap > 0:
            id_map[new_id] = best_old_id
            used_old_ids.add(best_old_id)

    # Assign fresh IDs to unmapped new communities. Fresh IDs avoid
    # collision with any previous-known ID and any already-assigned
    # fresh ID.
    reserved_ids = set(previous_partition.values()) | set(used_old_ids) | set(id_map.values())
    next_fresh = max(reserved_ids, default=-1) + 1
    for new_id in _stable_iteration(new_buckets.keys()):
        if new_id not in id_map:
            while next_fresh in reserved_ids:
                next_fresh += 1
            id_map[new_id] = next_fresh
            reserved_ids.add(next_fresh)

    remapped = {node: id_map[community] for node, community in new_partition.items()}
    return remapped, id_map


def _stable_iteration(items: Iterable[int]) -> list[int]:
    """Sorted iteration over IDs so fresh-ID assignment is deterministic."""
    return sorted(items)
