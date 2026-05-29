# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared colour palette for graph + visualisation surfaces.

Thermall power-station palette: sage / rust / apricot / cream on a
dark background. Used by viz renderers (pyvis, D3) and any future
HTML report templates so a corpus has a consistent visual identity
across surfaces. Previously duplicated across viz scripts; single
source now.
"""

from __future__ import annotations

# Node colour per entity type. Sage = documents (primary), warm tones
# for the bibliographic surface (authors, citations), cool tones for
# the conceptual surface (topics, methods, named_entities).
TYPE_COLORS: dict[str, str] = {
    "document": "#79c39e",  # sage primary
    "author": "#e77843",  # rust
    "citation": "#ee9b69",  # apricot
    "topic": "#ead1b5",  # cream
    "method": "#a8d5ba",  # sage tint
    "named_entity": "#f4b08e",  # apricot tint
    "gene": "#f4b08e",  # apricot tint
    "person": "#e77843",  # rust
    "org": "#ee9b69",  # apricot
    "place": "#d9c7a7",  # cream tint
    "other": "#999999",
    "unknown": "#666666",
}

# Edge colour per relation. Citation + authorship edges are saturated
# so they stand out against the dominant co_mentioned_in noise floor.
RELATION_COLORS: dict[str, str] = {
    "co_mentioned_in": "rgba(255,255,255,0.04)",
    "cites": "rgba(231,120,67,0.35)",
    "authored_by": "rgba(121,195,158,0.35)",
    "mentions": "rgba(168,213,186,0.20)",
    "shares_summary_with": "rgba(238,155,105,0.30)",
    "has_topic": "rgba(234,209,181,0.30)",
}
