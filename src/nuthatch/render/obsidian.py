# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Obsidian vault export for a corpus.

Purpose: write a `<corpus>/cards/`, `<corpus>/communities/`,
    `<corpus>/dashboard.md`, `<corpus>/index.md`, and
    `<corpus>/log.md` set so the corpus opens as a navigable
    Obsidian vault. Dataview queries in `dashboard.md` and
    `index.md` let users filter by tag, year, community, etc.

Inputs at `export_vault`: a `CorpusLayout` + the data to render
    (per-paper metadata dicts and per-community membership maps).

Outputs: files written under `<corpus>/`. Logs the export to
    `<corpus>/log.md` (append-only, matching PhD KB's log format).

Pattern reused from `~/PhD-knowledge-base/knowledge_base/` vault
layout (subdirs per content type + top-level dashboard.md /
index.md / log.md with Dataview queries). The per-paper card is
rendered by `nuthatch.render.card.render_card` (separate module).
The per-community page renderer is new: no PhD KB equivalent
because PhD KB does not cluster.

Assumptions: the caller has already produced the per-paper
    metadata and the community partition. This module orchestrates
    file writes, not extraction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nuthatch.corpus.layout import CorpusLayout
from nuthatch.render.card import render_card


# Catppuccin Mocha named accents, ordered so adjacent community ids
# land on contrasting hues. Same ordering the D3 viz uses, so cards
# and the HTML graph share a colour identity per community.
# Catppuccin Mocha named accents — must stay byte-identical to the
# `CATPPUCCIN` array in `viz/d3_renderer.py` so the same community id
# renders the same colour in the Obsidian graph view and in the D3
# topology HTML. If you change one, change the other.
_CATPPUCCIN_HEX: tuple[str, ...] = (
    "#89B4FA",  # blue
    "#FAB387",  # peach
    "#A6E3A1",  # green
    "#F38BA8",  # pink
    "#94E2D5",  # teal
    "#F9E2AF",  # yellow
    "#CBA6F7",  # mauve
    "#EBA0AC",  # maroon
    "#74C7EC",  # sapphire
    "#F5C2E7",  # flamingo
    "#179299",  # Catppuccin Latte teal (replaces lavender — deeper
                # teal that reads better against the dark bg and
                # avoids confusion with the mauve at slot 6)
    "#89DCEB",  # sky
    "#F5E0DC",  # rosewater
    "#A6ADC8",  # subtext1
)


def _slugify(text: str, max_len: int = 64) -> str:
    """Lowercase, dashed, ASCII-safe slug for filenames + tags.

    Drops anything that isn't alpha-numeric, collapses runs of
    separators, trims edges. Caps at `max_len` so Obsidian doesn't
    refuse the filename on long topical labels.
    """
    if not text:
        return "untitled"
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if not s:
        return "untitled"
    return s[:max_len].rstrip("-")


def _hex_to_obsidian_rgb(hex_color: str) -> int:
    """Obsidian's graph.json stores colors as a single packed int.

    Format: (R << 16) | (G << 8) | B. Same encoding Obsidian's UI
    writes when the user picks a color via the colour picker.
    """
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) << 16) | (int(h[2:4], 16) << 8) | int(h[4:6], 16)


def _community_color(cid: int) -> str:
    """Return the Catppuccin hex for a community id, with wrap."""
    return _CATPPUCCIN_HEX[cid % len(_CATPPUCCIN_HEX)]


def _build_obsidian_graph_config(
    community_slugs: Mapping[int, str],
) -> dict[str, Any]:
    """Build the .obsidian/graph.json payload with one colour group
    per community. Queries against `tag:#cluster/<cid>`, the neutral
    membership marker injected on each card (the topical label lives
    on the community page itself, not on member tags)."""
    groups = []
    for cid in sorted(community_slugs):
        groups.append({
            "query": f"tag:#cluster/{cid}",
            "color": {
                "a": 1,
                "rgb": _hex_to_obsidian_rgb(_community_color(cid)),
            },
        })
    # Field values tuned from active use on a comparable nuthatch-
    # adjacent KB (PhD knowledge-base). Panels expand by default so
    # the user sees groups + display options on first open; arrows
    # are on so directed edges read correctly; physics tweaked for
    # graphs in the 1k–4k node range.
    #
    # `showTags: True` keeps Obsidian's tag pseudo-nodes visible in
    # the graph view. The default theme paints them lime-green; the
    # ship `.obsidian/snippets/nuthatch-graph-colors.css` snippet
    # (auto-enabled via appearance.json) overrides that to the Power
    # Station accent orange so they read as a deliberate "tag" colour
    # alongside the per-community cluster fills.
    return {
        "collapse-filter": False,
        "search": "",
        "showTags": True,
        "showAttachments": False,
        "hideUnresolved": False,
        "showOrphans": True,
        "collapse-color-groups": False,
        "colorGroups": groups,
        "collapse-display": False,
        "showArrow": True,
        "textFadeMultiplier": 0,
        "nodeSizeMultiplier": 1.26,
        "lineSizeMultiplier": 0.81,
        "collapse-forces": False,
        "centerStrength": 0.42,
        "repelStrength": 12.3,
        "linkStrength": 1,
        "linkDistance": 99,
        "scale": 0.21,
        "close": True,
    }


def _write_obsidian_graph(root: Path, community_slugs: Mapping[int, str]) -> None:
    """Drop .obsidian/graph.json at the vault root. Non-destructive:
    only writes the graph view config, leaves other .obsidian files
    (workspace.json, app.json, ...) untouched."""
    obsidian_dir = root / ".obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    path = obsidian_dir / "graph.json"
    path.write_text(
        json.dumps(_build_obsidian_graph_config(community_slugs), indent=2),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ExportResult:
    """Files written by `export_vault`."""

    n_cards: int
    n_communities: int
    dashboard_path: Path
    index_path: Path
    log_path: Path
    cards_dir: Path
    communities_dir: Path


def export_vault(
    layout: CorpusLayout,
    *,
    paper_metadata: Mapping[str, Mapping[str, Any]],
    community_membership: Mapping[str, list[str]] | None = None,
    community_descriptions: Mapping[str, str] | None = None,
    source_note: str = "ingest",
) -> ExportResult:
    """Write the per-corpus vault: cards, communities, dashboard, index, log."""
    cards_dir = layout.root / "cards"
    communities_dir = layout.root / "communities"
    cards_dir.mkdir(parents=True, exist_ok=True)
    communities_dir.mkdir(parents=True, exist_ok=True)

    # Load the persisted community index if cluster stage has run.
    # When present, every card gets community_id + community_path +
    # community_label injected into its frontmatter so agents can
    # navigate community-aware retrieval directly from the card.
    # Fall back through known backend filenames when the generic
    # communities.json alias has not been written.
    from nuthatch.clustering.persist import load_community_index

    community_index = load_community_index(layout)
    if community_index is None:
        for _fname in (
            "communities_sbm.json",
            "communities_leiden.json",
            "communities_embeddings.json",
        ):
            community_index = load_community_index(layout, index_filename=_fname)
            if community_index is not None:
                break

    # Precompute community_id -> slug mapping once. Slugs drive:
    #   - community page filenames (e.g. communities/genomic-interactions.md
    #     instead of communities/23.md, so the Obsidian graph view shows
    #     readable node names instead of bare numbers)
    #   - the `community/<slug>` tag injected on each card, which
    #     `.obsidian/graph.json` colour groups query against
    community_slugs: dict[int, str] = {}
    if community_index:
        for cid_int, raw in community_index.labels.items():
            if isinstance(raw, str) and raw.startswith("doc::"):
                rep_doc = raw[len("doc::"):]
                label = str(
                    paper_metadata.get(rep_doc, {}).get("title") or raw
                )
            else:
                label = str(raw) if raw else f"community-{cid_int}"
            community_slugs[int(cid_int)] = _slugify(label)

    n_cards = 0
    for doc_id, meta in paper_metadata.items():
        cid = community_index.community_for(doc_id) if community_index else None
        cpath = community_index.hierarchy_for(doc_id) if community_index else None
        clabel_raw = (
            community_index.labels.get(cid)
            if community_index and cid is not None
            else None
        )
        # labels stored as doc_ids — resolve to paper title when possible
        if clabel_raw and clabel_raw.startswith("doc::"):
            rep_doc = clabel_raw[len("doc::"):]
            clabel = str(paper_metadata.get(rep_doc, {}).get("title") or clabel_raw)
        else:
            clabel = clabel_raw

        # Build merged tag list: topic tags from the extractor PLUS a
        # neutral `cluster/<cid>` membership marker (numeric id, no
        # topical claim). The topical label lives on the community page
        # only; applying it as a tag here would mis-label cluster members
        # whose paper isn't actually about the cluster's headline theme
        # (heterogeneous clusters happen and are normal). Obsidian
        # graph.json colour groups query `tag:#cluster/<cid>`.
        #
        # Slugify each tag — Obsidian rejects tag values containing
        # whitespace (e.g. "electricity price forecasting"), so the
        # extractor's free-text topics need to become dashed slugs
        # before they go in the `tags:` array. The human-readable form
        # survives in the separate `topics:` array on each card.
        raw_topics = list(meta.get("topics") or meta.get("tags") or [])
        base_tags = [_slugify(str(t)) for t in raw_topics if t]
        if cid is not None:
            base_tags.append(f"cluster/{cid}")

        card_md = render_card(
            doc_id=doc_id,
            metadata=dict(meta),
            tags=base_tags,
            community_id=cid,
            community_path=cpath if cpath else None,
            community_label=clabel,
        )
        (cards_dir / f"{doc_id}.md").write_text(card_md, encoding="utf-8")
        n_cards += 1

    # Wipe stale community pages from any prior render (e.g. numeric
    # `23.md` files from the era before slug-based filenames). Done
    # before the write loop so re-runs converge to the current label
    # set instead of accumulating files for dropped community ids.
    if community_membership:
        for stale in communities_dir.glob("*.md"):
            stale.unlink()

    n_communities = 0
    if community_membership:
        for community_id, members in community_membership.items():
            description = (
                community_descriptions.get(community_id, "")
                if community_descriptions
                else ""
            )
            cid_int = int(community_id) if community_index else None
            clabel_page_raw = (
                community_index.labels.get(cid_int)
                if community_index and cid_int is not None
                else None
            )
            if clabel_page_raw and clabel_page_raw.startswith("doc::"):
                rep_doc = clabel_page_raw[len("doc::"):]
                clabel_page = str(
                    paper_metadata.get(rep_doc, {}).get("title") or clabel_page_raw
                )
            else:
                clabel_page = clabel_page_raw
            core_ids_raw = (
                community_index.core_nodes_of(cid_int)
                if community_index and cid_int is not None
                else []
            )
            core_ids = [
                n[len("doc::"):] if n.startswith("doc::") else n
                for n in core_ids_raw
            ]
            page = _render_community_page(
                community_id=community_id,
                member_doc_ids=members,
                paper_metadata=paper_metadata,
                description=description,
                community_label=clabel_page,
                core_doc_ids=core_ids or None,
                backend=community_index.backend if community_index else None,
            )
            # Filename uses the topical slug so Obsidian's graph view
            # shows a readable node name instead of the bare community id.
            try:
                slug = community_slugs.get(int(community_id)) or community_id
            except (TypeError, ValueError):
                slug = community_id
            (communities_dir / f"{slug}.md").write_text(
                page, encoding="utf-8"
            )
            n_communities += 1

    # Drop .obsidian/graph.json with one colour group per community so
    # the native Obsidian graph view colourizes nodes by community on
    # first vault open. Non-destructive to other Obsidian config.
    if community_slugs:
        _write_obsidian_graph(layout.root, community_slugs)

    dashboard_path = layout.root / "dashboard.md"
    dashboard_path.write_text(_render_dashboard(), encoding="utf-8")

    index_path = layout.root / "index.md"
    index_path.write_text(_render_index(), encoding="utf-8")

    log_path = layout.root / "log.md"
    _append_log(
        log_path,
        source_note=source_note,
        n_cards=n_cards,
        n_communities=n_communities,
    )

    return ExportResult(
        n_cards=n_cards,
        n_communities=n_communities,
        dashboard_path=dashboard_path,
        index_path=index_path,
        log_path=log_path,
        cards_dir=cards_dir,
        communities_dir=communities_dir,
    )


def _render_community_page(
    *,
    community_id: str,
    member_doc_ids: list[str],
    paper_metadata: Mapping[str, Mapping[str, Any]],
    description: str,
    community_label: str | None = None,
    core_doc_ids: list[str] | None = None,
    backend: str | None = None,
) -> str:
    """One markdown page per community. Links to member cards."""
    display_title = community_label or f"Community {community_id}"
    tags = ["community"]
    if backend:
        tags.append(backend.replace("_", "-"))
    tags_yaml = "[" + ", ".join(tags) + "]"
    lines = [
        "---",
        f"title: \"{display_title}\"",
        f"id: community_{community_id}",
        "type: community",
        f"community_id: {community_id}",
        f"n_members: {len(member_doc_ids)}",
        f"tags: {tags_yaml}",
    ]
    if backend:
        lines.append(f"backend: {backend}")
    lines += ["---", "", f"# {display_title}", ""]
    if description:
        lines.extend([description.strip(), ""])
    if core_doc_ids:
        lines.extend(["## Core papers", ""])
        for doc_id in core_doc_ids:
            meta = paper_metadata.get(doc_id, {})
            title = str(meta.get("title") or doc_id)
            year = meta.get("year")
            suffix = f" ({year})" if year else ""
            lines.append(f"- [[{doc_id}|{title}{suffix}]]")
        lines.append("")
    lines.append("## Members")
    lines.append("")
    for doc_id in member_doc_ids:
        meta = paper_metadata.get(doc_id, {})
        title = str(meta.get("title") or doc_id)
        year = meta.get("year")
        suffix = f" ({year})" if year else ""
        lines.append(f"- [[{doc_id}|{title}{suffix}]]")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_dashboard() -> str:
    """High-level dashboard with Dataview queries.

    Pattern reused from PhD KB `dashboard.md`: a small set of
    targeted TABLE / LIST queries the user runs in Obsidian. Each
    block here is a starting point; users edit per-corpus.
    """
    return """---
title: Corpus Dashboard
description: Fast access patterns for the active nuthatch corpus
tags: [dashboard]
---

# Corpus Dashboard

> Use Cmd+G to open the graph view, or run the Dataview queries below
> for filtered lookups. Each query block is a starting point; edit
> per your corpus's tag vocabulary.

## Recently ingested papers

```dataview
TABLE year, authors, ingested
FROM "cards"
WHERE type = "paper"
SORT ingested DESC
LIMIT 25
```

## Highest-relevance papers

```dataview
TABLE relevance, year, authors
FROM "cards"
WHERE type = "paper"
SORT relevance DESC
LIMIT 25
```

## Papers by community

```dataview
TABLE community_id, year, authors
FROM "cards"
WHERE type = "paper" AND community_id != null
SORT community_id ASC, year ASC
```

## Communities

```dataview
TABLE n_members
FROM "communities"
SORT n_members DESC
```

## Pending review (status = exploratory)

```dataview
TABLE year, authors, ingested
FROM "cards"
WHERE status = "exploratory"
SORT ingested ASC
```
"""


def _render_index() -> str:
    """Vault index with content catalog Dataview tables.

    Pattern reused from PhD KB `index.md`: one TABLE per content
    type so the user can jump to any sub-collection.
    """
    return """---
title: Corpus Index
description: Content catalog for the nuthatch corpus
tags: [index]
---

# Corpus Index

## Papers

```dataview
TABLE title, year, authors
FROM "cards"
WHERE type = "paper"
SORT year ASC, title ASC
```

## Communities

```dataview
TABLE community_id, n_members
FROM "communities"
SORT community_id ASC
```

## Tags

```dataview
LIST
FROM "cards"
FLATTEN tags as tag
GROUP BY tag
SORT tag ASC
```
"""


def _append_log(
    log_path: Path,
    *,
    source_note: str,
    n_cards: int,
    n_communities: int,
) -> None:
    """Append a timestamped entry to the corpus log.

    Format mirrors the PhD KB `log.md` convention: HTML-comment
    marker for entry shape, newest entries at the top.
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    entry = (
        f"\n## {timestamp} -- {source_note}\n"
        f"**Cards written:** {n_cards}\n"
        f"**Community pages written:** {n_communities}\n\n"
    )
    if log_path.is_file():
        content = log_path.read_text(encoding="utf-8")
        marker = "-->\n"
        if marker in content:
            idx = content.index(marker) + len(marker)
            content = content[:idx] + entry + content[idx:]
        else:
            content += entry
    else:
        content = (
            "---\n"
            "title: Corpus Log\n"
            "description: Append-only chronological record of "
            "ingests and exports\n"
            "---\n\n"
            "# Corpus Log\n\n"
            "<!-- Append new entries at the top. Format:\n"
            "## YYYY-MM-DD HH:MM -- [ingest|export|maintenance]\n"
            "**Cards written:** N\n"
            "**Community pages written:** M\n"
            "**Notes:** any observations\n"
            "-->\n"
            + entry
        )
    log_path.write_text(content, encoding="utf-8")
