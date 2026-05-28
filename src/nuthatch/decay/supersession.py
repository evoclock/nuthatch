# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Supersession scanner + edge writer.

Purpose: when a card declares `supersedes: [<old_doc_id>, ...]` in
    its frontmatter, the decay pass must:

    1. Add a `supersedes` edge from the new doc node to each old
       doc node in the corpus graph.
    2. Flip the OLD card's `status` to `superseded` (was probably
       `exploratory` or `validated`).
    3. Downweight the OLD doc's `relevance` by a fixed multiplier
       so it drops out of the default Dataview views in Obsidian
       and the LLM-context retriever.

The convention comes from kb-reports.md:

    > supersedes: IDs this report *replaces*. Writing
    > `supersedes: [X]` flips X's `status` to `superseded`
    > (convention; enforced by decay pass or linter, future work).

This module is the "decay pass" half of that promise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx

_DEFAULT_PREDECESSOR_WEIGHT: float = 0.1


@dataclass(frozen=True)
class SupersessionPair:
    """One supersedes declaration: `new_doc_id` replaces `old_doc_id`."""

    new_doc_id: str
    old_doc_id: str


@dataclass(frozen=True)
class SupersessionResult:
    pairs_applied: list[SupersessionPair]
    pairs_skipped: list[tuple[SupersessionPair, str]]  # (pair, reason)


def find_supersession_pairs(
    cards: Mapping[str, Mapping[str, Any]],
) -> list[SupersessionPair]:
    """Walk `cards` (`doc_id -> frontmatter`) and collect supersession pairs.

    Each card's `supersedes` field is either a list of doc IDs or
    absent. Pairs whose old_doc_id is not in `cards` are still
    returned; the caller decides whether to skip them at apply
    time.
    """
    pairs: list[SupersessionPair] = []
    for new_doc_id, front in cards.items():
        raw = front.get("supersedes")
        if not raw:
            continue
        if isinstance(raw, str):
            raw = [raw]
        for old in raw:
            old_str = str(old).strip()
            if old_str:
                pairs.append(SupersessionPair(new_doc_id=new_doc_id, old_doc_id=old_str))
    return pairs


def apply_supersession(
    g: nx.MultiDiGraph,
    cards: dict[str, dict[str, Any]],
    pairs: list[SupersessionPair],
    *,
    predecessor_weight: float = _DEFAULT_PREDECESSOR_WEIGHT,
) -> SupersessionResult:
    """Add `supersedes` edges + flip old cards' status + downweight.

    Mutates `g` and `cards` in place. Pairs whose old_doc_id has no
    card in the corpus are skipped with a reason (the new doc may
    cite an external prior work that nuthatch doesn't have).
    """
    applied: list[SupersessionPair] = []
    skipped: list[tuple[SupersessionPair, str]] = []
    for pair in pairs:
        old_card = cards.get(pair.old_doc_id)
        if old_card is None:
            skipped.append((pair, "old doc not in corpus"))
            continue
        new_node_id = f"doc::{pair.new_doc_id}"
        old_node_id = f"doc::{pair.old_doc_id}"
        # Add the edge even if the graph hasn't seen the doc node yet
        # (NetworkX creates nodes implicitly on add_edge); the
        # downstream renderer + graph io tolerate this.
        g.add_edge(
            new_node_id,
            old_node_id,
            relation="supersedes",
            confidence="EXTRACTED",
        )
        old_card["status"] = "superseded"
        current_relevance = float(old_card.get("relevance", 1.0))
        old_card["relevance"] = round(current_relevance * predecessor_weight, 4)
        applied.append(pair)
    return SupersessionResult(pairs_applied=applied, pairs_skipped=skipped)
