# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Relevance decay for graph nodes.

Purpose: weight nodes by recency + connectivity so the LLM surface
    and the clustering layer can demote stale, unconnected papers
    without deleting them. Formula from
    `~/project-planning-agent/conventions/kb-reports.md`:

        relevance(t) = max(backlinks, 1) * exp(-ln2 * Δt / half_life)

    `Δt` is days since `last_touched`. `half_life` is the per-node
    `half_life_days` attribute (default 365 for papers). `backlinks`
    is the count of edges INTO the node.

Inputs at `decay_score`: a node's age in days + backlink count +
    half-life. Outputs a relevance score in (0, ∞).

Inputs at `apply_decay`: a `MultiDiGraph` + a "now" date. Walks
    every node and updates its `relevance` attribute.

Assumptions: nodes carry `last_touched` (ISO date string) and
    `half_life_days` (int or None) attributes. Nodes with
    `half_life_days = None` are pinned and never decay.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import networkx as nx

_LN2: float = math.log(2.0)
_DEFAULT_PAPER_HALF_LIFE: int = 365


def decay_score(
    *,
    days_since_touched: float,
    backlinks: int,
    half_life_days: int | None,
) -> float:
    """Compute the decay-weighted relevance for one node.

    `half_life_days = None` short-circuits to a high constant
    (the node is pinned and not subject to decay).
    """
    if half_life_days is None:
        return float(max(backlinks, 1))
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive when not None")
    if days_since_touched < 0:
        days_since_touched = 0.0
    attenuation = math.exp(-_LN2 * days_since_touched / float(half_life_days))
    return float(max(backlinks, 1)) * attenuation


def _parse_iso_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            return None


def apply_decay(
    g: nx.MultiDiGraph,
    *,
    now: date | None = None,
    default_half_life_days: int = _DEFAULT_PAPER_HALF_LIFE,
) -> nx.MultiDiGraph:
    """Walk `g` and update every node's `relevance` attribute.

    Returns the same graph instance (mutated in place).
    """
    today = now or date.today()
    for node_id, attrs in g.nodes(data=True):
        touched = _parse_iso_date(attrs.get("last_touched"))
        # Treat untouched nodes as zero-age so they don't get silently
        # decayed when the date field is missing.
        days = 0.0 if touched is None else float((today - touched).days)

        half_life_raw = attrs.get("half_life_days", default_half_life_days)
        half_life: int | None
        if half_life_raw is None:
            half_life = None
        else:
            try:
                half_life = int(half_life_raw)
            except (TypeError, ValueError):
                half_life = default_half_life_days

        backlinks = g.in_degree(node_id)
        attrs["relevance"] = decay_score(
            days_since_touched=days,
            backlinks=backlinks,
            half_life_days=half_life,
        )
    return g
