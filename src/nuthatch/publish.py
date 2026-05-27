# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Promote a corpus's agent-facing surface into a shareable KB directory.

Where `render` produces cards/communities/dashboards inside the corpus
directory (intermixed with PDFs and working state), `publish` exports
only the **consumable** surface to a destination of the operator's
choice. The destination is the artifact you ship — a folder that an
agent host can `nuthatch serve` against, or a human can open as an
Obsidian vault, or commit to its own git repo for distribution.

Two roles, two directories:

    ~/my-corpus/           working corpus (PDFs, .kg/extracted/, audit)
    ~/my-kb-name/          published KB (this module's output)

What lands in the destination by default:

    Human-navigable (Obsidian):
        README.md, AGENTS.md, OVERVIEW.md, LICENSE
        cards/, communities/, dashboard.md, index.md, log.md
        .obsidian/{graph,app,appearance,community-plugins}.json
        docs/graph.html  (D3 topology viz if available)

    Agent-readable (MCP):
        .kg/graph/graph.json
        .kg/communities_*.json
        .kg/community_centroids.npy (powers community_search)
        .kg/embeddings/  (chroma store — optional, opt out via flag)
        CAPABILITIES.json, mcp_config.example.json

    Scientific provenance (if available):
        docs/eval-*.md
        publish_manifest.jsonl

What is NEVER copied:

    Source PDFs (copyright)
    .kg/extracted/*.md  (full body text; opt in with --include-bodies)
    .kg/audit/  (operator-private logs)
    .kg/manifest.jsonl  (corpus-internal paths)

Re-running `publish` against the same destination is idempotent and
overwrites generated artifacts. The publish_manifest.jsonl records
per-file provenance so a re-publish is diffable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

# Sane defaults for license. CC-BY-4.0 is the right call for a corpus
# derived from CC-BY-4.0 preprints (typical for bioRxiv / arXiv mixes).
# Operator can override via --license <SPDX-id> or "PROPRIETARY".
_DEFAULT_LICENSE_SPDX = "CC-BY-4.0"

_LICENSE_TEMPLATES: dict[str, str] = {
    "CC-BY-4.0": (
        "This work is licensed under the Creative Commons "
        "Attribution 4.0 International License (CC-BY-4.0).\n\n"
        "Full text: https://creativecommons.org/licenses/by/4.0/\n\n"
        "You are free to share and adapt the material with appropriate "
        "attribution.\n\n"
        "NOTE: This license applies to the knowledge base structure "
        "(cards, community pages, graph, indexes) produced by nuthatch. "
        "Original source documents (papers, preprints) remain under "
        "their respective licenses; their full text is not included in "
        "this distribution by default.\n"
    ),
    "CC-BY-SA-4.0": (
        "This work is licensed under the Creative Commons "
        "Attribution-ShareAlike 4.0 International License (CC-BY-SA-4.0).\n\n"
        "Full text: https://creativecommons.org/licenses/by-sa/4.0/\n"
    ),
    "CC0-1.0": (
        "This work is dedicated to the public domain under the Creative "
        "Commons CC0 1.0 Universal Public Domain Dedication.\n\n"
        "Full text: https://creativecommons.org/publicdomain/zero/1.0/\n"
    ),
    "Apache-2.0": (
        "This work is licensed under the Apache License, Version 2.0.\n\n"
        "Full text: https://www.apache.org/licenses/LICENSE-2.0\n"
    ),
    "MIT": (
        "MIT License\n\n"
        "Permission is hereby granted, free of charge, to any person "
        "obtaining a copy of this material to deal in it without "
        "restriction.\n"
    ),
    "PROPRIETARY": (
        "ALL RIGHTS RESERVED.\n\n"
        "This knowledge base is distributed for the recipient's "
        "internal use only. No license is granted for redistribution, "
        "modification, or sublicensing without explicit written "
        "permission from the copyright holder.\n"
    ),
}


@dataclass(frozen=True)
class PublishResult:
    """What `publish_corpus` produced."""

    dest: Path
    n_cards: int
    n_community_pages: int
    bytes_written: int
    included_chroma: bool
    included_bodies: bool
    included_eval: bool
    manifest_path: Path


def _now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path, chunk: int = 65536) -> str:
    """Per-file SHA-256 for the publish manifest. Skips files that
    error on read (returns 'unreadable') so a single bad file doesn't
    abort the whole publish."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while block := f.read(chunk):
                h.update(block)
    except OSError:
        return "unreadable"
    return h.hexdigest()


def _rel_src(src: Path, corpus_root: Path) -> str:
    """Render a source path as relative-to-corpus so the published
    manifest doesn't leak the operator's absolute filesystem layout.
    Falls back to the basename for paths outside the corpus (rare;
    only happens for tool-repo eval reports etc.)."""
    try:
        return str(src.resolve().relative_to(corpus_root))
    except ValueError:
        return src.name


def _copy_tree(
    src_dir: Path,
    dest_dir: Path,
    role: str,
    manifest: list[dict[str, Any]],
    root: Path,
    corpus_root: Path,
    include_glob: str = "*",
) -> tuple[int, int]:
    """Mirror a directory tree into the destination. Returns
    (n_files, bytes_total). Manifest stores src as a corpus-relative
    path (or basename if outside the corpus)."""
    if not src_dir.is_dir():
        return 0, 0
    n_files = 0
    total_bytes = 0
    for src in src_dir.rglob(include_glob):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir)
        dest = dest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        size = dest.stat().st_size
        manifest.append(
            {
                "dest": str(dest.relative_to(root)),
                "src": _rel_src(src, corpus_root),
                "role": role,
                "size": size,
                "sha256": _file_sha256(dest),
                "promoted_at": _now_utc_iso(),
            }
        )
        n_files += 1
        total_bytes += size
    return n_files, total_bytes


def _write_text(
    dest: Path, content: str, role: str, manifest: list[dict[str, Any]], root: Path
) -> int:
    """Write a generated text file and record it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    size = dest.stat().st_size
    manifest.append(
        {
            "dest": str(dest.relative_to(root)),
            "src": "<generated>",
            "role": role,
            "size": size,
            "sha256": _file_sha256(dest),
            "promoted_at": _now_utc_iso(),
        }
    )
    return size


# ---------- Generated content ---------------------------------------


def _readme(
    kb_name: str, n_cards: int, n_communities: int, community_labels: dict[int, str]
) -> str:
    """Top-level vault README. The first thing a human sees on clone."""
    top_communities = "\n".join(
        f"- **{label}** (`communities/`)" for _, label in sorted(community_labels.items())
    )
    return f"""# {kb_name}

A nuthatch-published knowledge base.

This directory is a self-contained artifact: open it as an Obsidian vault to
navigate visually, or point an MCP-aware agent at it for programmatic
retrieval. Nothing here requires a build step — every index, page, and
config is pre-rendered.

## After cloning: unpack the vector store

The ChromaDB vector store ships compressed as `embeddings.tar.gz`
at the repo root (~60 MB) so this repo pushes through plain git
without LFS quotas. Unpack once after cloning:

```bash
mkdir -p .kg && tar xzf embeddings.tar.gz -C .kg/
```

That creates `.kg/embeddings/`, which `nuthatch serve` reads
directly. The `.tar.gz` can be deleted after extraction if you
want to save disk; re-clone to recover.

## Quick start (human)

1. Install [Obsidian](https://obsidian.md).
2. **File → Open folder as vault**, select this directory.
3. **Settings → Community plugins**, install **Dataview**, enable it.
4. Open `dashboard.md` for the entry point. The graph view
   (`Ctrl+G`/`Cmd+G`) colourizes nodes by community automatically
   (`.obsidian/graph.json` ships pre-configured colour rules).

## Quick start (agent)

After unpacking the vector store (above), if you have nuthatch
installed (`pipx install nuthatch`):

```bash
nuthatch serve --corpus .
```

This exposes the MCP query surface over stdio JSON-RPC. Drop
`mcp_config.example.json` into your agent host's MCP config file
to register it (see that file for per-host instructions).

Without nuthatch installed, the chroma store under `.kg/embeddings/`
is still readable by any MCP-compatible vector retriever, and the
graph + community indexes are plain JSON under `.kg/`.

## Structure

```text
.
├── cards/              {n_cards} per-paper Obsidian cards (one .md per doc)
├── communities/        {n_communities} per-community topical pages
├── dashboard.md        Dataview entry point (filtered tables)
├── index.md            Content catalog
├── log.md              Append-only ingest/publish history
├── docs/
│   ├── graph.html      Interactive D3 topology viz (open in browser)
│   └── eval-*.md       Extraction / cluster / retrieval quality reports
├── .obsidian/          Pre-configured vault settings + colour groups
├── .kg/                Agent-readable indexes
│   ├── graph/graph.json        Corpus graph (nodes + typed edges)
│   ├── communities_*.json      Community partitions (one per backend)
│   ├── community_centroids.npy Powers semantic search at the community level
│   └── embeddings/             ChromaDB vector store
├── AGENTS.md           Agent-facing description + MCP tool semantics
├── CAPABILITIES.json   Machine-readable feature/index inventory
└── publish_manifest.jsonl  Per-file provenance + sha256
```

## Top communities

{top_communities}

See `communities/` for the full list and `OVERVIEW.md` for stats.

## Provenance

This KB was built with [nuthatch](https://github.com/evoclock/nuthatch).
See `publish_manifest.jsonl` for per-file source + checksum, and
`OVERVIEW.md` for build parameters + corpus stats. The eval reports
under `docs/` document extraction precision, cluster quality, and
retrieval performance against this corpus.

## License

See `LICENSE`.
"""


def _agents_md(
    kb_name: str, n_cards: int, n_communities: int, community_labels: dict[int, str]
) -> str:
    """KB-root AGENTS.md. Tells an agent what's available, the card
    schema, and how to register the MCP server."""
    community_list = "\n".join(
        f"- `{cid}` — {label}" for cid, label in sorted(community_labels.items())
    )
    return f"""---
name: {kb_name}
description: nuthatch-published knowledge base ({n_cards} documents, {n_communities} communities). Local-first, agent-readable, MCP-served.
trigger: nuthatch serve --corpus <path-to-this-directory>
---

# {kb_name} — agent contract

This is a nuthatch-published knowledge base. It carries:

- A document graph: documents + extracted entities (authors, citations,
  topics, methods) with typed edges (authored_by, cites, about, mentions_*).
- Communities: a partition of documents by topical/structural similarity.
  Each community has a representative label and an in-graph centroid.
- A vector store (ChromaDB) over chunked document text for dense retrieval.

## MCP tools

When served via `nuthatch serve --corpus <path>`, the following tools
are exposed over stdio JSON-RPC:

| Tool | Purpose |
| --- | --- |
| `corpus_search(query, k)` | Chunk-level dense retrieval. Hits include `community_id`, `community_path`, `community_label` so you can route into community tools without an extra round-trip. |
| `subgraph_extract(seed_nodes, depth)` | BFS subgraph around seeds. |
| `card_get(doc_id)` | Full per-doc card markdown (frontmatter has community fields). |
| `community_get(community_id)` | Full per-community page markdown. |
| `community_brief(community_id, top_n)` | Cheap structured preamble (label, n_members, top-N representatives) before paying card-fetch cost. |
| `community_search(query, k)` | Semantic search at the community level — ranks communities by query-to-centroid cosine. Jump straight to the relevant cluster. |
| `community_core_nodes(community_id)` | High-degree members within the community. The "key papers" of the cluster. |
| `community_hierarchy(doc_id)` | Walk the nested SBM hierarchy (leaf → super-communities). |
| `token_econ_report(group_by, since, until)` | Aggregate per-tool counterfactual savings. |

The community tools are the headline — they let you do graph-RAG over the
corpus without paying full-card costs to learn community membership.
Typical flow:

```text
1. community_search("relevant topic", k=3)
   → three community_ids ranked by semantic match
2. community_brief(community_id, top_n=5) on the winner
   → label + 5 representative doc_ids, ~200 tokens
3. drill into one card (card_get) or read the whole community page
   (community_get) or walk the parent community (community_hierarchy)
```

## Card frontmatter contract

Every file under `cards/` carries this YAML frontmatter:

```yaml
title: <human-readable paper title>
id: <doc_id>
type: paper
date: <ISO date when card was rendered>
status: exploratory          # or validated, superseded, deprecated, pinned
tags:                        # Obsidian-valid slugified tags
  - <topic-slug-1>
  - ...
  - cluster/<community_id>   # neutral membership marker
doc_id: <doc_id>
authors: [...]
year: <int>
doi: <string-or-empty>
arxiv_id: <string-or-empty>
topics: [<topic-1>, ...]      # human-readable topics (the source of `tags`)
community_id: <int>            # leaf community
community_path: [<leaf>, ..., <root>]  # SBM nested chain (single-element for flat backends)
community_label: <topical name of the community>
relevance: <float 0..1>
half_life_days: <int>
ingested: <UTC iso>
last_touched: <UTC iso>
n_chunks: <int>                # how many chunks the body was split into
```

The `cluster/<community_id>` tag is **a membership marker, not a topic
claim**. The community's topical label belongs to the community as a
whole; individual member papers may not all be about that exact topic
(heterogeneous clusters happen and are normal).

## Community page contract

Files under `communities/` are named by the topical slug:

```yaml
title: "<topical name>"
id: community_<numeric_id>
type: community
community_id: <int>
n_members: <int>
tags: [community, <backend-name>]
backend: <sbm | leiden | embeddings>
```

The body lists core papers (highest-degree members within the community
subgraph) and the full membership as Obsidian wikilinks to cards.

## Indexes available

Inspect `CAPABILITIES.json` for a machine-readable list. At a glance:

- `.kg/graph/graph.json` — corpus graph as JSON node-link
- `.kg/communities_sbm.json` (and sibling backends if present) — partition + labels + hierarchy
- `.kg/community_centroids.npy` — float32 matrix [n_communities, embedding_dim], indexed by `community_ids` from the matching JSON
- `.kg/embeddings/` — ChromaDB store (sqlite-backed)

## Communities in this KB

{community_list}
"""


def _overview(
    kb_name: str,
    n_cards: int,
    n_communities: int,
    community_labels: dict[int, str],
    graph_stats: dict[str, int],
    backends_present: list[str],
) -> str:
    """Auto-generated stats overview."""
    backends = ", ".join(f"`{b}`" for b in backends_present) or "(none)"
    return (
        f"""# {kb_name} — overview

Auto-generated at publish time. For a navigable view see `README.md`;
for an agent contract see `AGENTS.md`.

## Build

- **Tool**: [nuthatch](https://github.com/evoclock/nuthatch)
- **Published**: {_now_utc_iso()}
- **Clustering backends present**: {backends}

## Corpus stats

| metric | value |
| --- | ---: |
| Documents (cards) | {n_cards} |
| Communities | {n_communities} |
| Graph nodes (docs + entities) | {graph_stats.get("nodes", 0)} |
| Graph edges (typed) | {graph_stats.get("edges", 0)} |

## Communities

| id | label |
| ---: | --- |
"""
        + "\n".join(f"| {cid} | {label} |" for cid, label in sorted(community_labels.items()))
        + "\n"
    )


def _capabilities_json(
    n_cards: int,
    n_communities: int,
    graph_stats: dict[str, int],
    backends_present: list[str],
    has_centroids: bool,
    has_chroma: bool,
) -> str:
    """Machine-readable feature inventory."""
    payload = {
        "schema_version": 1,
        "tool": "nuthatch",
        "kb_kind": "nuthatch-published",
        "built_at": _now_utc_iso(),
        "capabilities": {
            "vector_search": has_chroma,
            "graph_traversal": True,
            "community_routing": True,
            "community_search": has_centroids,
            "community_hierarchy": True,
            "card_fetch": True,
            "token_economy": True,
        },
        "indexes": {
            "graph": {"present": True, "path": ".kg/graph/graph.json"},
            "communities": {
                "present": bool(backends_present),
                "backends": backends_present,
                "path_template": ".kg/communities_<backend>.json",
                "canonical_path": ".kg/communities.json",
            },
            "community_centroids": {
                "present": has_centroids,
                "path": ".kg/community_centroids.npy",
            },
            "chroma": {
                "present": has_chroma,
                "path": ".kg/embeddings/",
            },
        },
        "stats": {
            "n_cards": n_cards,
            "n_communities": n_communities,
            "n_graph_nodes": graph_stats.get("nodes", 0),
            "n_graph_edges": graph_stats.get("edges", 0),
        },
    }
    return json.dumps(payload, indent=2) + "\n"


def _mcp_config_example(kb_name: str) -> str:
    """Drop-in MCP server config snippet. Comments embedded as JSON
    string keys aren't valid; we emit a JSON-with-instructions
    wrapper so the user can paste the inner `mcpServers` block into
    their host config."""
    cfg = {
        "_instructions": (
            "Paste the contents of `mcpServers` into your host's MCP "
            "config file. Replace <ABSOLUTE-PATH-TO-THIS-KB> with the "
            "fully-qualified path. For Claude Code: ~/.claude.json. "
            "For Codex / OpenCode / Aider / Pi / Hermes: see "
            "agent_skills/skill-<host>.md in the nuthatch tool repo."
        ),
        "mcpServers": {
            f"nuthatch-{kb_name}": {
                "command": "nuthatch",
                "args": ["serve", "--corpus", "<ABSOLUTE-PATH-TO-THIS-KB>"],
            }
        },
    }
    return json.dumps(cfg, indent=2) + "\n"


def _obsidian_app_json() -> str:
    return (
        json.dumps(
            {
                "useTab": False,
                "tabSize": 2,
                "showLineNumber": False,
                "showInlineTitle": True,
                "showViewHeader": True,
                "livePreview": True,
                "readableLineLength": True,
                "defaultViewMode": "preview",
                "newLinkFormat": "shortest",
            },
            indent=2,
        )
        + "\n"
    )


def _obsidian_appearance_json() -> str:
    """Power Station palette accents on Obsidian's default dark theme.
    Enables the nuthatch-graph-colors CSS snippet so tag pseudo-nodes
    in the graph view paint orange instead of the default lime-green."""
    return (
        json.dumps(
            {
                "accentColor": "#e77843",
                "theme": "obsidian",
                "baseFontSize": 16,
                "showInlineTitle": True,
                "enabledCssSnippets": ["nuthatch-graph-colors"],
            },
            indent=2,
        )
        + "\n"
    )


def _nuthatch_graph_colors_css() -> str:
    """CSS snippet that overrides the graph-view tag pseudo-node colour
    to the Power Station accent orange. Obsidian's colour-groups
    target files (markdown content) and can't reach the synthetic
    tag nodes the graph view creates from card frontmatter; only a
    CSS variable override does. Covers both modern Obsidian (CSS
    var) and older versions (selector class) so it works across
    releases."""
    return """/*
 * nuthatch-graph-colors
 *
 * Repaints Obsidian's tag pseudo-nodes in the graph view to the
 * Power Station accent orange (#e77843). Without this snippet, tag
 * pseudo-nodes pick up the theme default (typically lime-green),
 * which clashes with the per-community Catppuccin palette used for
 * card nodes via .obsidian/graph.json colour groups.
 *
 * Auto-enabled by .obsidian/appearance.json's enabledCssSnippets.
 */

/* Modern Obsidian uses CSS variables for graph node colours. */
body {
    --graph-node-tag: #e77843;
}

/* Older Obsidian / fallback selector. Harmless when the variable
   above already applies; lets the snippet keep working across
   versions without depending on the user's Obsidian release. */
.graph-view.color-tag,
.theme-dark .graph-view.color-tag,
.theme-light .graph-view.color-tag {
    color: #e77843;
}
"""


def _obsidian_core_plugins_json() -> str:
    """Enable Obsidian's built-in graph + tag-pane plugins so the
    vault is immediately useful on first open."""
    return (
        json.dumps(
            [
                "file-explorer",
                "global-search",
                "switcher",
                "graph",
                "backlink",
                "outgoing-link",
                "tag-pane",
                "page-preview",
                "templates",
                "note-composer",
                "command-palette",
                "outline",
                "word-count",
                "starred",
            ],
            indent=2,
        )
        + "\n"
    )


def _obsidian_community_plugins_json() -> str:
    """The dashboard's Dataview queries require this plugin. Obsidian
    will prompt the user to install it on first open (community
    plugins aren't bundled in distribution per Obsidian's terms)."""
    return json.dumps(["dataview"], indent=2) + "\n"


# ---------- The publish driver ---------------------------------------


def publish_corpus(
    *,
    corpus_root: Path,
    dest: Path,
    kb_name: str | None = None,
    license_spdx: str = _DEFAULT_LICENSE_SPDX,
    include_chroma: bool = True,
    include_bodies: bool = False,
    include_eval: bool = True,
    include_d3_html: bool = True,
    tool_repo_root: Path | None = None,
) -> PublishResult:
    """Promote a corpus's agent-facing surface into `dest`.

    `corpus_root` is the directory the operator ran the pipeline
    against (the one with `.kg/`, `cards/`, etc.). `dest` is where
    the published KB lands; it is created if missing, and any files
    we own are overwritten (publish is idempotent).

    `tool_repo_root` is optional and only used to copy `docs/eval-*.md`
    and `docs/architecture/` if available; pass the nuthatch source
    repo root, or omit to skip those.
    """
    corpus_root = corpus_root.resolve()
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    kb_name = kb_name or dest.name

    manifest: list[dict[str, Any]] = [{"_root": str(dest)}]
    total_bytes = 0

    # ----- Navigable surface (Obsidian) ---------------------------
    n_cards = 0
    cards_src = corpus_root / "cards"
    if cards_src.is_dir():
        n, b = _copy_tree(cards_src, dest / "cards", "card", manifest, dest, corpus_root, "*.md")
        n_cards = n
        total_bytes += b

    n_community_pages = 0
    comm_src = corpus_root / "communities"
    if comm_src.is_dir():
        n, b = _copy_tree(
            comm_src, dest / "communities", "community-page", manifest, dest, corpus_root, "*.md"
        )
        n_community_pages = n
        total_bytes += b

    for top in ("dashboard.md", "index.md", "log.md"):
        src = corpus_root / top
        if src.is_file():
            total_bytes += _copy_file_recorded(
                src,
                dest / top,
                "navigation",
                manifest,
                dest,
                corpus_root,
            )

    # ----- Obsidian config --------------------------------------------
    obsidian_dest = dest / ".obsidian"
    obsidian_dest.mkdir(parents=True, exist_ok=True)

    # graph.json is rendered by render/obsidian.py into the corpus
    # already; carry it forward. If missing, we still write app/
    # appearance/plugin defaults so the vault is usable.
    obsidian_src = corpus_root / ".obsidian" / "graph.json"
    if obsidian_src.is_file():
        total_bytes += _copy_file_recorded(
            obsidian_src,
            obsidian_dest / "graph.json",
            "obsidian-config",
            manifest,
            dest,
            corpus_root,
        )

    total_bytes += _write_text(
        obsidian_dest / "app.json",
        _obsidian_app_json(),
        "obsidian-config",
        manifest,
        dest,
    )
    total_bytes += _write_text(
        obsidian_dest / "appearance.json",
        _obsidian_appearance_json(),
        "obsidian-config",
        manifest,
        dest,
    )
    total_bytes += _write_text(
        obsidian_dest / "core-plugins.json",
        _obsidian_core_plugins_json(),
        "obsidian-config",
        manifest,
        dest,
    )
    total_bytes += _write_text(
        obsidian_dest / "community-plugins.json",
        _obsidian_community_plugins_json(),
        "obsidian-config",
        manifest,
        dest,
    )
    total_bytes += _write_text(
        obsidian_dest / "snippets" / "nuthatch-graph-colors.css",
        _nuthatch_graph_colors_css(),
        "obsidian-snippet",
        manifest,
        dest,
    )

    # ----- Agent-readable indexes -------------------------------------
    kg_src = corpus_root / ".kg"
    kg_dest = dest / ".kg"
    backends_present: list[str] = []
    has_centroids = False
    if kg_src.is_dir():
        # Community indexes. Copy these BEFORE the graph so the merge
        # step has community labels available. Canonical first (the
        # glob below requires an underscore after `communities` and
        # therefore does not match the canonical name); then every
        # suffixed sibling. Order matters for the defensive
        # promotion below: an existing canonical at source must
        # always win over a suffixed promotion.
        canonical_src = kg_src / "communities.json"
        if canonical_src.is_file():
            total_bytes += _copy_file_recorded(
                canonical_src,
                kg_dest / canonical_src.name,
                "community-index-canonical",
                manifest,
                dest,
                corpus_root,
            )
        for cj in sorted(kg_src.glob("communities_*.json")):
            backend = cj.stem.removeprefix("communities_")
            backends_present.append(backend)
            total_bytes += _copy_file_recorded(
                cj,
                kg_dest / cj.name,
                "community-index",
                manifest,
                dest,
                corpus_root,
            )

        # Graph — merge community per node from the SBM partition so
        # downstream consumers (D3 viz, external tools, agents) get a
        # self-contained data structure. This is the same shape
        # graphify ships in `graphify-out/graph.json` (per-node
        # `community: <int>`) and means the `.obsidian/graph.json`
        # colour groups + this file + the D3 viz chip legend all
        # agree on community ids → same Catppuccin colour per cluster.
        graph_src = kg_src / "graph" / "graph.json"
        sbm_src = kg_src / "communities_sbm.json"
        if graph_src.is_file():
            graph_dest_path = kg_dest / "graph" / "graph.json"
            graph_dest_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                merged = _merge_community_into_graph(graph_src, sbm_src)
                graph_dest_path.write_text(
                    json.dumps(merged, indent=2),
                    encoding="utf-8",
                )
                size = graph_dest_path.stat().st_size
                manifest.append(
                    {
                        "dest": str(graph_dest_path.relative_to(dest)),
                        "src": _rel_src(graph_src, corpus_root),
                        "role": "graph-index-merged",
                        "size": size,
                        "sha256": _file_sha256(graph_dest_path),
                        "promoted_at": _now_utc_iso(),
                    }
                )
                total_bytes += size
            except (json.JSONDecodeError, OSError):
                # Fall back to a plain copy if merging fails for any
                # reason; never break publish over a graph-shape edge case.
                total_bytes += _copy_file_recorded(
                    graph_src,
                    graph_dest_path,
                    "graph-index",
                    manifest,
                    dest,
                    corpus_root,
                )
        # Centroids: canonical name first, then any suffixed copies.
        cent_src = kg_src / "community_centroids.npy"
        if cent_src.is_file():
            has_centroids = True
            total_bytes += _copy_file_recorded(
                cent_src,
                kg_dest / "community_centroids.npy",
                "centroids",
                manifest,
                dest,
                corpus_root,
            )
        for cent_suffixed in sorted(kg_src.glob("community_centroids_*.npy")):
            total_bytes += _copy_file_recorded(
                cent_suffixed,
                kg_dest / cent_suffixed.name,
                "centroids",
                manifest,
                dest,
                corpus_root,
            )

        # Defensive canonical promotion: the MCP server's community_*
        # tools look for `communities.json` and
        # `community_centroids.npy` (unsuffixed). When the source
        # corpus was clustered with only `--output-suffix` runs (so
        # the canonical names were never produced), promote a
        # preferred backend to the canonical name at the publish dest
        # so the published KB is self-contained. Preference order:
        # sbm > leiden > embeddings (sbm carries the nested hierarchy
        # the other backends do not). This is dest-only; the source
        # corpus is never touched.
        canonical_index_dest = kg_dest / "communities.json"
        if not canonical_index_dest.is_file():
            for backend in ("sbm", "leiden", "embeddings"):
                cand = kg_dest / f"communities_{backend}.json"
                if cand.is_file():
                    shutil.copy2(cand, canonical_index_dest)
                    size = canonical_index_dest.stat().st_size
                    manifest.append(
                        {
                            "dest": str(canonical_index_dest.relative_to(dest)),
                            "src": f"(promoted from communities_{backend}.json)",
                            "role": "community-index-canonical",
                            "size": size,
                            "sha256": _file_sha256(canonical_index_dest),
                            "promoted_at": _now_utc_iso(),
                        }
                    )
                    total_bytes += size
                    break

        canonical_cent_dest = kg_dest / "community_centroids.npy"
        if not canonical_cent_dest.is_file():
            for backend in ("sbm", "leiden", "embeddings"):
                cand = kg_dest / f"community_centroids_{backend}.npy"
                if cand.is_file():
                    shutil.copy2(cand, canonical_cent_dest)
                    has_centroids = True
                    size = canonical_cent_dest.stat().st_size
                    manifest.append(
                        {
                            "dest": str(canonical_cent_dest.relative_to(dest)),
                            "src": f"(promoted from community_centroids_{backend}.npy)",
                            "role": "centroids-canonical",
                            "size": size,
                            "sha256": _file_sha256(canonical_cent_dest),
                            "promoted_at": _now_utc_iso(),
                        }
                    )
                    total_bytes += size
                    break

    has_chroma = False
    if include_chroma:
        chroma_src = kg_src / "embeddings"
        if chroma_src.is_dir() and any(chroma_src.iterdir()):
            # Ship the chroma store as a single .tar.gz at the KB
            # ROOT (not under .kg/) so users see the file they need
            # to unpack — anything under .kg/ is hidden on Unix-like
            # filesystems and easy to miss. Unpack target stays
            # `.kg/embeddings/`, which is where `nuthatch serve`
            # looks. Plain-git friendly: under GitHub's 100 MB
            # per-file limit, no git-lfs bandwidth quota. gzip
            # gives ~37% reduction on the sqlite + HNSW indexes
            # (94 MB → ~59 MB on a typical 100-doc corpus).
            archive_path = dest / "embeddings.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(chroma_src, arcname="embeddings")
            size = archive_path.stat().st_size
            manifest.append(
                {
                    "dest": str(archive_path.relative_to(dest)),
                    "src": _rel_src(chroma_src, corpus_root),
                    "role": "chroma-archive",
                    "size": size,
                    "sha256": _file_sha256(archive_path),
                    "promoted_at": _now_utc_iso(),
                }
            )
            total_bytes += size
            has_chroma = True

    if include_bodies:
        ext_src = kg_src / "extracted"
        if ext_src.is_dir():
            n, b = _copy_tree(
                ext_src,
                kg_dest / "extracted",
                "extracted-body",
                manifest,
                dest,
                corpus_root,
                "*.md",
            )
            total_bytes += b
            # Also copy the per-doc .meta.json sidecars
            n, b = _copy_tree(
                ext_src,
                kg_dest / "extracted",
                "extracted-meta",
                manifest,
                dest,
                corpus_root,
                "*.meta.json",
            )
            total_bytes += b

    # ----- D3 graph viz (if available) --------------------------------
    if include_d3_html:
        pipe_out = corpus_root.parent / "pipeline_output"
        if not pipe_out.is_dir() and tool_repo_root is not None:
            pipe_out = tool_repo_root / "pipeline_output"
        if pipe_out.is_dir():
            latest = max(
                pipe_out.glob("graph_topology_d3_sbm_*.html"),
                default=None,
                key=lambda p: p.stat().st_mtime,
            )
            if latest is not None:
                total_bytes += _copy_file_recorded(
                    latest,
                    dest / "docs" / "graph.html",
                    "graph-viz",
                    manifest,
                    dest,
                    corpus_root,
                )

    # ----- Eval reports (if available) --------------------------------
    if include_eval and tool_repo_root is not None:
        eval_src_dir = tool_repo_root / "docs"
        if eval_src_dir.is_dir():
            for eval_md in sorted(eval_src_dir.glob("eval-*.md")):
                total_bytes += _copy_file_recorded(
                    eval_md,
                    dest / "docs" / eval_md.name,
                    "eval-report",
                    manifest,
                    dest,
                    corpus_root,
                )

    # ----- Community labels (for generated content) -------------------
    community_labels: dict[int, str] = {}
    sbm_index = kg_src / "communities_sbm.json"
    if sbm_index.is_file():
        try:
            data = json.loads(sbm_index.read_text(encoding="utf-8"))
            for cid_str, raw in (data.get("labels") or {}).items():
                community_labels[int(cid_str)] = str(raw)
        except (json.JSONDecodeError, ValueError):
            pass

    # ----- Graph stats (for generated content) ------------------------
    # nuthatch persists graph.json with a `{schema_version, graph_type,
    # data}` wrapper, where `data` carries the networkx node-link shape
    # (`nodes` + `links`). Older snapshots stored it un-wrapped at top
    # level, so fall through.
    graph_stats: dict[str, int] = {"nodes": 0, "edges": 0}
    graph_path = kg_dest / "graph" / "graph.json"
    if graph_path.is_file():
        try:
            g = json.loads(graph_path.read_text(encoding="utf-8"))
            payload = g.get("data") if isinstance(g.get("data"), dict) else g
            graph_stats["nodes"] = len(payload.get("nodes", []))
            graph_stats["edges"] = len(payload.get("links", payload.get("edges", [])))
        except json.JSONDecodeError:
            pass

    # ----- Generated content (README, AGENTS, OVERVIEW, etc) ----------
    total_bytes += _write_text(
        dest / "README.md",
        _readme(kb_name, n_cards, n_community_pages, community_labels),
        "generated",
        manifest,
        dest,
    )
    total_bytes += _write_text(
        dest / "AGENTS.md",
        _agents_md(kb_name, n_cards, n_community_pages, community_labels),
        "generated",
        manifest,
        dest,
    )
    total_bytes += _write_text(
        dest / "OVERVIEW.md",
        _overview(
            kb_name, n_cards, n_community_pages, community_labels, graph_stats, backends_present
        ),
        "generated",
        manifest,
        dest,
    )
    total_bytes += _write_text(
        dest / "CAPABILITIES.json",
        _capabilities_json(
            n_cards, n_community_pages, graph_stats, backends_present, has_centroids, has_chroma
        ),
        "generated",
        manifest,
        dest,
    )
    total_bytes += _write_text(
        dest / "mcp_config.example.json",
        _mcp_config_example(kb_name),
        "generated",
        manifest,
        dest,
    )
    total_bytes += _write_text(
        dest / "LICENSE",
        _LICENSE_TEMPLATES.get(license_spdx, _LICENSE_TEMPLATES[_DEFAULT_LICENSE_SPDX]),
        "license",
        manifest,
        dest,
    )

    # ----- Publish manifest -------------------------------------------
    # Strip the sentinel _root entry before writing.
    manifest_records = [m for m in manifest if "_root" not in m]
    manifest_path = dest / "publish_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    return PublishResult(
        dest=dest,
        n_cards=n_cards,
        n_community_pages=n_community_pages,
        bytes_written=total_bytes,
        included_chroma=has_chroma,
        included_bodies=include_bodies,
        included_eval=include_eval and (tool_repo_root is not None),
        manifest_path=manifest_path,
    )


def _copy_file_recorded(
    src: Path, dest: Path, role: str, manifest: list[dict[str, Any]], root: Path, corpus_root: Path
) -> int:
    """Single-file copy that records into the publish manifest. `src`
    is stored as corpus-relative so the published manifest doesn't
    leak the operator's absolute filesystem layout."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    size = dest.stat().st_size
    manifest.append(
        {
            "dest": str(dest.relative_to(root)),
            "src": _rel_src(src, corpus_root),
            "role": role,
            "size": size,
            "sha256": _file_sha256(dest),
            "promoted_at": _now_utc_iso(),
        }
    )
    return size


def _merge_community_into_graph(
    graph_path: Path,
    sbm_index_path: Path,
) -> dict[str, Any]:
    """Read `graph.json`, merge `community` + `community_label` onto
    each document node from the SBM partition, return the merged dict
    ready to write back. Idempotent: re-merging is fine, fields just
    get overwritten with the latest partition.

    nuthatch persists the graph in two shapes:
      - new: `{schema_version, graph_type, data: {nodes, edges}}`
      - legacy: `{nodes, links}` at top level
    Handle both.
    """
    graph: dict[str, Any] = cast(dict[str, Any], json.loads(graph_path.read_text(encoding="utf-8")))
    _inner = graph.get("data")
    data: dict[str, Any] = _inner if isinstance(_inner, dict) else graph

    # Build doc_id -> (community_id, community_label) from SBM if present.
    cid_by_doc: dict[str, int] = {}
    labels_by_cid: dict[int, str] = {}
    if sbm_index_path.is_file():
        try:
            sbm = json.loads(sbm_index_path.read_text(encoding="utf-8"))
            for k, v in (sbm.get("flat") or {}).items():
                # Persisted keys may carry `doc::` prefix or be bare.
                bare = k[len("doc::") :] if k.startswith("doc::") else k
                cid_by_doc[bare] = int(v)
                # Also accept lookups by the full prefixed form.
                cid_by_doc[f"doc::{bare}"] = int(v)
            for k, v in (sbm.get("labels") or {}).items():
                labels_by_cid[int(k)] = str(v)
        except (json.JSONDecodeError, ValueError):
            pass

    for node in data.get("nodes", []):
        nid = node.get("id")
        if nid is None:
            continue
        # Only doc nodes get a community assignment (entity nodes
        # don't participate in the document-level partition).
        if not (str(nid).startswith("doc::") or node.get("node_type") == "document"):
            continue
        cid = cid_by_doc.get(str(nid))
        if cid is None:
            continue
        node["community"] = cid
        label = labels_by_cid.get(cid)
        if label:
            node["community_label"] = label

    return graph
