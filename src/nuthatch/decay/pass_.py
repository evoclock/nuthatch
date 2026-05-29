# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end decay pass orchestrator.

The pass:

1. Load the corpus graph from `.kg/graph/graph.json`.
2. Walk every `cards/<doc_id>.md`, parse frontmatter.
3. Push frontmatter `last_touched`, `half_life_days`, and any
   `status: pinned` onto matching graph nodes so the decay kernel
   has authoritative values.
4. Apply supersession scan + edges + status flips + downweight on
   the old cards.
5. Run `graph.decay.apply_decay` (in-place).
6. Pull the post-decay `relevance` back onto card frontmatter
   (with today's `last_touched` only when the value actually
   changed, so re-running the pass on an unchanged corpus is a
   no-op for `last_touched`).
7. Optionally write the graph back. Write cards back unless
   `dry_run=True`.

Idempotency: a second consecutive run over an unchanged corpus
produces the same outputs (the rounding tolerance + the
"unchanged value -> don't bump last_touched" rule guarantee it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import networkx as nx

from nuthatch.corpus.layout import CorpusLayout
from nuthatch.decay.frontmatter import parse_card, write_card
from nuthatch.decay.supersession import (
    apply_supersession,
    find_supersession_pairs,
)
from nuthatch.graph.decay import apply_decay
from nuthatch.graph.io import load_graph, save_graph

_DEFAULT_ARCHIVE_THRESHOLD: float = 0.1
_RELEVANCE_EQUAL_TOLERANCE: float = 1e-4


@dataclass(frozen=True)
class ArchiveCandidate:
    doc_id: str
    relevance: float
    backlinks: int
    last_touched: str | None


@dataclass(frozen=True)
class SupersessionEvent:
    new_doc_id: str
    old_doc_id: str
    applied: bool
    reason: str | None = None  # None when applied=True


@dataclass(frozen=True)
class DecayResult:
    n_cards_scanned: int
    n_decayed: int
    n_pinned: int
    n_superseded_this_pass: int
    archive_candidates: list[ArchiveCandidate]
    supersession_events: list[SupersessionEvent]
    dry_run: bool
    pass_date: str
    archive_threshold: float = _DEFAULT_ARCHIVE_THRESHOLD
    cards_written: int = 0
    graph_written: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


def run_decay_pass(
    layout: CorpusLayout,
    *,
    dry_run: bool = False,
    now: date | None = None,
    archive_threshold: float = _DEFAULT_ARCHIVE_THRESHOLD,
) -> DecayResult:
    """Run a full decay + supersession pass on a corpus."""
    today = now or datetime.now(UTC).date()
    cards_dir = layout.root / "cards"
    graph_path = layout.kg / "graph" / "graph.json"

    # 1. Load graph (or empty MultiDiGraph if the corpus has none).
    g = load_graph(graph_path) if graph_path.is_file() else nx.MultiDiGraph()

    # 2. Walk cards. Parse frontmatter; remember body for write-back.
    cards: dict[str, dict[str, Any]] = {}
    bodies: dict[str, str] = {}
    paths: dict[str, Path] = {}
    if cards_dir.is_dir():
        for card_path in sorted(cards_dir.glob("*.md")):
            front, body = parse_card(card_path)
            if not front:
                continue
            doc_id = str(front.get("doc_id") or front.get("id") or card_path.stem)
            cards[doc_id] = front
            bodies[doc_id] = body
            paths[doc_id] = card_path

    # 3. Sync card-side authoritative fields onto graph nodes.
    for doc_id, front in cards.items():
        node_id = f"doc::{doc_id}"
        if node_id not in g:
            g.add_node(node_id, node_type="document")
        attrs = g.nodes[node_id]
        if front.get("last_touched") is not None:
            attrs["last_touched"] = front["last_touched"]
        if "half_life_days" in front:
            attrs["half_life_days"] = front["half_life_days"]
        # `status: pinned` overrides any non-None half_life so the
        # decay kernel treats it as immutable.
        if str(front.get("status", "")).lower() == "pinned":
            attrs["half_life_days"] = None

    # 4. Run the decay kernel first (mutates node `relevance` in place).
    apply_decay(g, now=today)

    # 5. Pull updated relevance back onto card frontmatter.
    n_decayed = 0
    n_pinned = 0
    cards_to_write: list[str] = []
    for doc_id, front in cards.items():
        node_id = f"doc::{doc_id}"
        if node_id not in g:
            continue
        new_relevance = g.nodes[node_id].get("relevance")
        if new_relevance is None:
            continue
        old_relevance = float(front.get("relevance", 1.0))
        if str(front.get("status", "")).lower() == "pinned":
            n_pinned += 1
            # Pinned cards: kernel still wrote a relevance (the
            # max(backlinks,1) value); accept it but don't bump
            # last_touched for it specifically — pinned means stable.
            continue
        if abs(float(new_relevance) - old_relevance) > _RELEVANCE_EQUAL_TOLERANCE:
            front["relevance"] = round(float(new_relevance), 4)
            # NB: `last_touched` is NOT bumped here. Per kb-reports.md
            # `last_touched` means "last human edit", not "last machine
            # update". Bumping it on every decay pass would reset
            # `days_since_touched` to 0, undoing the decay on the next
            # run (idempotency violation).
            n_decayed += 1
            cards_to_write.append(doc_id)

    # 6. Supersession runs AFTER decay. The downweight multiplies the
    # post-decay relevance, so a superseded card's final relevance is
    # (decayed * predecessor_weight). Edges + status flips also happen
    # here. Order matters: doing supersession before decay would let
    # the kernel overwrite the downweight in step 4.
    pairs = find_supersession_pairs(cards)
    super_result = apply_supersession(g, cards, pairs)
    supersession_events = [
        SupersessionEvent(new_doc_id=p.new_doc_id, old_doc_id=p.old_doc_id, applied=True)
        for p in super_result.pairs_applied
    ] + [
        SupersessionEvent(
            new_doc_id=p.new_doc_id,
            old_doc_id=p.old_doc_id,
            applied=False,
            reason=reason,
        )
        for p, reason in super_result.pairs_skipped
    ]
    # Supersession-flipped cards must be written too even if their
    # relevance change was below the decay tolerance.
    for pair in super_result.pairs_applied:
        if pair.old_doc_id in cards and pair.old_doc_id not in cards_to_write:
            cards_to_write.append(pair.old_doc_id)

    # 7. Archive-candidate scan (post-decay).
    archive_candidates: list[ArchiveCandidate] = []
    for doc_id, front in cards.items():
        relevance = float(front.get("relevance", 1.0))
        if relevance < archive_threshold:
            node_id = f"doc::{doc_id}"
            backlinks = g.in_degree(node_id) if node_id in g else 0
            archive_candidates.append(
                ArchiveCandidate(
                    doc_id=doc_id,
                    relevance=round(relevance, 4),
                    backlinks=backlinks,
                    last_touched=str(front.get("last_touched") or ""),
                )
            )
    archive_candidates.sort(key=lambda c: c.relevance)

    # Writes (skipped on dry-run).
    cards_written = 0
    graph_written = False
    if not dry_run:
        for doc_id in cards_to_write:
            write_card(paths[doc_id], cards[doc_id], bodies[doc_id])
            cards_written += 1
        if cards:
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            save_graph(g, graph_path)
            graph_written = True

    return DecayResult(
        n_cards_scanned=len(cards),
        n_decayed=n_decayed,
        n_pinned=n_pinned,
        n_superseded_this_pass=len(super_result.pairs_applied),
        archive_candidates=archive_candidates,
        supersession_events=supersession_events,
        dry_run=dry_run,
        pass_date=today.isoformat(),
        archive_threshold=archive_threshold,
        cards_written=cards_written,
        graph_written=graph_written,
    )


def render_report(result: DecayResult, *, corpus_name: str | None = None) -> str:
    """Render a markdown summary of the decay pass for human reading."""
    name_line = f" — {corpus_name}" if corpus_name else ""
    lines = [
        f"# Decay pass report{name_line}",
        f"_{result.pass_date}_",
        "",
        "## Summary",
        "",
        f"- Cards scanned: {result.n_cards_scanned}",
        f"- Cards decayed (relevance updated): {result.n_decayed}",
        f"- Cards pinned (no decay): {result.n_pinned}",
        f"- Supersession events this pass: {result.n_superseded_this_pass}",
        f"- Archive candidates (relevance < {result.archive_threshold}): "
        f"{len(result.archive_candidates)}",
        f"- Mode: {'dry-run (no writes)' if result.dry_run else 'committed'}",
    ]
    if not result.dry_run:
        lines.append(f"- Cards written: {result.cards_written}")
        lines.append(f"- Graph written: {result.graph_written}")
    lines.append("")
    lines.append("## Archive candidates")
    lines.append("")
    if not result.archive_candidates:
        lines.append("_none_")
    else:
        lines.append("| doc_id | relevance | backlinks | last_touched |")
        lines.append("| --- | ---: | ---: | --- |")
        for c in result.archive_candidates:
            lines.append(
                f"| {c.doc_id} | {c.relevance} | {c.backlinks} | {c.last_touched or '-'} |"
            )
    lines.append("")
    lines.append("## Supersession events")
    lines.append("")
    if not result.supersession_events:
        lines.append("_none_")
    else:
        for ev in result.supersession_events:
            if ev.applied:
                lines.append(f"- {ev.new_doc_id} supersedes {ev.old_doc_id}")
            else:
                lines.append(f"- skipped: {ev.new_doc_id} supersedes {ev.old_doc_id} ({ev.reason})")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
