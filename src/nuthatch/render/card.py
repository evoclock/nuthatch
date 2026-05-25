# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Per-paper markdown card renderer.

Purpose: render the per-paper card that lands at
    `<corpus>/cards/<doc_id>.md` after ingest. The card carries
    Dataview-queryable YAML frontmatter and a body of structured
    sections (Authors, DOI, Abstract, Key Claims, Methods, Topics,
    Relevance) that Obsidian renders and the clustering layer
    consumes.

Inputs: `doc_id`, extracted metadata dict, optional excerpt and
    chunk count.

Outputs: a markdown string ready to write under
    `<corpus>/cards/<doc_id>.md`.

Pattern reused from `~/PhD-knowledge-base/scripts/02_assemble_wiki_page.py`
(slugify + section-by-section body assembly + YAML frontmatter via
`yaml.dump`). nuthatch's implementation is a fresh write; the
frontmatter schema aligns with the **kb-reports.md frontmatter
contract** in `~/project-planning-agent/conventions/kb-reports.md`
so Obsidian Dataview queries and the clustering layer can index
nuthatch cards alongside PhD KB reports and proposal notes with a
single vocabulary.

Frontmatter fields (paper-specific specialisation of the kb-reports
contract):

- `title`, `id`, `date` (required by kb-reports)
- `type: paper` (distinguishes from `work_report` / `concept`)
- `doc_id`, `authors`, `year`, `doi`, `arxiv_id` (paper-specific)
- `topics`, `committee_member`, `aim_ref` (kb-reports tags)
- `status` (kb-reports vocabulary: exploratory / validated / ...)
- `relevance` (default 1.0, recomputed by future decay pass)
- `half_life_days` (default 365 for papers; per kb-reports schema)
- `ingested`, `last_touched` (operational timestamps)
- `n_chunks` (set by the ingest pipeline)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Any

import yaml

# Paper cards get the kb-reports `validated`-style half-life (365 days)
# unless overridden, matching how manuscript work is treated. The
# half-life drives the relevance decay function from kb-reports.md.
_PAPER_HALF_LIFE_DAYS: int = 365


def slugify(text: str) -> str:
    """Slug helper from PhD KB pattern: lowercase, kebab/underscore-safe, length-capped.

    Used for the card filename and the `id` frontmatter field. Pattern
    reused from `~/PhD-knowledge-base/scripts/02_assemble_wiki_page.py`.
    """
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    return text.strip("_")[:80]


def _as_list(value: Any) -> list[Any]:
    """Coerce a scalar or list to a list (frontmatter likes lists)."""
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _today_iso() -> str:
    return date.today().isoformat()


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M")


def build_frontmatter(
    *,
    doc_id: str,
    metadata: dict[str, Any],
    n_chunks: int | None = None,
    tags: Iterable[str] | None = None,
    status: str = "exploratory",
    relevance: float = 1.0,
    half_life_days: int | None = _PAPER_HALF_LIFE_DAYS,
    aim_ref: str | None = None,
) -> dict[str, Any]:
    """Build the kb-reports-aligned frontmatter dict for a per-paper card.

    Field order matches the kb-reports.md contract so a human scanning
    the YAML sees the same shape across all knowledge-base artifacts.
    """
    title = str(metadata.get("title") or doc_id)
    front: dict[str, Any] = {
        "title": title,
        "id": doc_id,
        "type": "paper",
        "date": _today_iso(),
        "status": status,
        "tags": list(tags) if tags is not None else _as_list(metadata.get("topics")),
        "committee_member": _as_list(metadata.get("committee_member")),
        "aim_ref": aim_ref,
        "doc_id": doc_id,
        "authors": _as_list(metadata.get("authors")),
        "year": int(metadata["year"]) if metadata.get("year") else None,
        "doi": metadata.get("doi", ""),
        "arxiv_id": metadata.get("arxiv_id", ""),
        "topics": _as_list(metadata.get("topics")),
        "relevance": float(relevance),
        "half_life_days": half_life_days,
        "ingested": _now_iso(),
        "last_touched": _now_iso(),
        "n_chunks": n_chunks,
    }
    # Drop None-valued optional fields so the YAML stays clean.
    return {k: v for k, v in front.items() if v is not None and v != ""}


def render_card(
    *,
    doc_id: str,
    metadata: dict[str, Any],
    excerpt: str | None = None,
    n_chunks: int | None = None,
    tags: Iterable[str] | None = None,
    status: str = "exploratory",
    relevance: float = 1.0,
    half_life_days: int | None = _PAPER_HALF_LIFE_DAYS,
    aim_ref: str | None = None,
) -> str:
    """Render the full per-paper card markdown.

    The body section pattern is reused from PhD KB's
    `02_assemble_wiki_page.py`: Authors / Year / DOI header line,
    then Abstract, Key Claims, Methods, Topics (with Obsidian
    `[[Concept]]` links), and Relevance.
    """
    front = build_frontmatter(
        doc_id=doc_id,
        metadata=metadata,
        n_chunks=n_chunks,
        tags=tags,
        status=status,
        relevance=relevance,
        half_life_days=half_life_days,
        aim_ref=aim_ref,
    )
    yaml_text = yaml.dump(front, default_flow_style=False, sort_keys=False).rstrip()

    title = str(metadata.get("title") or doc_id)
    lines: list[str] = ["---", yaml_text, "---", "", f"# {title}", ""]

    authors = _as_list(metadata.get("authors"))
    if authors:
        lines.append(f"**Authors:** {', '.join(str(a) for a in authors)}")
    year = metadata.get("year")
    if year:
        lines.append(f"**Year:** {year}")
    doi = metadata.get("doi")
    if doi:
        lines.append(f"**DOI:** {doi}")
    arxiv_id = metadata.get("arxiv_id")
    if arxiv_id:
        lines.append(f"**arXiv:** {arxiv_id}")
    lines.append("")

    abstract = metadata.get("abstract")
    if abstract:
        lines.extend(["## Abstract", "", str(abstract).strip(), ""])

    key_claims = _as_list(metadata.get("key_claims"))
    if key_claims:
        lines.extend(["## Key Claims", ""])
        for claim in key_claims:
            lines.append(f"- {claim}")
        lines.append("")

    methods = _as_list(metadata.get("methods"))
    if methods:
        lines.extend(["## Methods", ""])
        for method in methods:
            lines.append(f"- {method}")
        lines.append("")

    topics = _as_list(metadata.get("topics"))
    if topics:
        lines.extend(["## Topics", ""])
        for t in topics:
            concept = str(t).replace("_", " ").title()
            lines.append(f"- [[{concept}]]")
        lines.append("")

    relevance_text = metadata.get("relevance_note")
    if relevance_text:
        lines.extend(["## Relevance", "", str(relevance_text).strip(), ""])

    if excerpt:
        lines.extend(["## Excerpt", "", excerpt.strip(), ""])

    return "\n".join(lines).rstrip() + "\n"
