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

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nuthatch.corpus.layout import CorpusLayout
from nuthatch.render.card import render_card


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

    n_cards = 0
    for doc_id, meta in paper_metadata.items():
        card_md = render_card(doc_id=doc_id, metadata=dict(meta))
        (cards_dir / f"{doc_id}.md").write_text(card_md, encoding="utf-8")
        n_cards += 1

    n_communities = 0
    if community_membership:
        for community_id, members in community_membership.items():
            description = (
                community_descriptions.get(community_id, "")
                if community_descriptions
                else ""
            )
            page = _render_community_page(
                community_id=community_id,
                member_doc_ids=members,
                paper_metadata=paper_metadata,
                description=description,
            )
            (communities_dir / f"{community_id}.md").write_text(
                page, encoding="utf-8"
            )
            n_communities += 1

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
) -> str:
    """One markdown page per community. Links to member cards."""
    lines = [
        "---",
        f"title: \"Community {community_id}\"",
        f"id: community_{community_id}",
        "type: community",
        f"community_id: {community_id}",
        f"n_members: {len(member_doc_ids)}",
        "tags: [community]",
        "---",
        "",
        f"# Community {community_id}",
        "",
    ]
    if description:
        lines.extend([description.strip(), ""])
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
> per your corpus tags and committee_members.

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
