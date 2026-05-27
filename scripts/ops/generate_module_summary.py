#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""
Script: generate_module_summary

Path: scripts/ops/generate_module_summary.py

Purpose: Produce a per-package per-script summary markdown from the
    repo-local `pipeline_output/codebase_inventory.jsonl`. Output is
    a Dataview-friendly section per package plus a top-level table.

Inputs: `pipeline_output/codebase_inventory.jsonl` (must exist; run
    `/inventory` from Claude Code if absent).

Outputs: stdout (redirect to docs/architecture/nuthatch_modules.md).

Assumptions: the inventory is current (PostToolUse hook keeps it
    fresh, but a manual `/inventory` may be needed if scripts were
    added during a fast-forward).

Parameters: none.

Failure Modes: missing inventory yields a clear error. Entries with
    no `module_docstring` are flagged in the output but not skipped.

Author: Julen Gamboa

Created: 2026-05-26

Last Edited: 2026-05-26 by Julen Gamboa
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _first_paragraph(text: str | None) -> str:
    """First docstring paragraph, collapsing whitespace, < 200 chars."""
    if not text:
        return "_no docstring_"
    # Skip the first line if it's a `Script: name` header; take the
    # `Purpose:` paragraph instead when present.
    for chunk in ("Purpose:", "purpose:"):
        if chunk in text:
            tail = text.split(chunk, 1)[1]
            para = tail.split("\n\n", 1)[0].strip()
            return _shorten(para)
    para = text.strip().split("\n\n", 1)[0]
    return _shorten(para)


def _shorten(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "..."


def _bucket(entry: dict) -> str | None:
    """Return the package name an entry belongs to, or None to skip."""
    p = Path(entry["path"])
    parts = p.parts
    if not parts:
        return None
    first = parts[0]
    if first == "src" and len(parts) >= 2 and parts[1] == "nuthatch":
        if len(parts) >= 4:
            return parts[2]
        return "_top"  # src/nuthatch/{cli.py, __init__.py}
    if first == "scripts":
        return f"scripts/{parts[1]}" if len(parts) >= 3 else "scripts"
    if first == "tests":
        return None  # tests omitted from the summary
    return first


_PACKAGE_BLURBS = {
    "_top": "Top-level entry points: CLI (`cli.py`) and the package `__init__`.",
    "ingest": "Stage 1: extract markdown from sources, validate schema, route to "
              "`processed/<subdir>/` or `quarantine/<reason>/`. Handles arxiv / "
              "bioRxiv metadata enrichment, math-retry flagging, dedup, "
              "orchestrator state machine, and the `triage` pre-flight "
              "(pdftotext-only PASS/FLAG/DEFER classification, exposed as "
              "`nuthatch triage` CLI subcommand).",
    "embed": "Stage 2: chunk extracted markdown and persist embeddings into "
             "Chroma (`.kg/embeddings/`). Hybrid chunker with full-doc coverage "
             "invariant; orchestrator handles incremental + `--force` re-embed.",
    "graph": "Stage 3: build the document graph from embeddings + co-citation + "
             "semantic similarity edges. Outputs to `graph/`.",
    "clustering": "Stage 4: community detection. SBM via graph-tool when "
                  "available (nested hierarchy), Leiden fallback (flat). Hub "
                  "exclusion + reattachment by majority neighbour. Stable "
                  "cluster IDs across re-runs. `persist.py` writes "
                  "`.kg/communities.json` + `.kg/community_centroids.npy` so "
                  "the MCP server's community tools can run without "
                  "re-clustering.",
    "render": "Stage 5: render the corpus as an Obsidian-compatible vault. "
              "Per-paper cards under `cards/`, community pages under "
              "`communities/`, plus top-level `dashboard.md`, `index.md`, "
              "`log.md`. Wikilinks between cards form the navigable graph "
              "Obsidian's graph view picks up automatically; Dataview "
              "queries in the dashboard filter by tag / year / community.",
    "schema": "Per-corpus metadata contracts. Profiles for arxiv, bioRxiv, "
              "internal docs, patents. Profile-router picks per-file by "
              "filename pattern.",
    "corpus": "Corpus discovery, layout, init, and registry. Defines the "
              "`processed/<subdir>/` and `quarantine/<reason>/` lifecycle.",
    "retrieve": "Query-side helpers used by the MCP server: BM25, Chroma vector "
                "search, reranker invocation, hybrid result merging.",
    "mcp": "Read-only MCP server exposing nine tools to agents: "
           "`corpus_search`, `subgraph_extract`, `card_get`, "
           "`community_get`, `community_brief`, `community_search`, "
           "`community_core_nodes`, `community_hierarchy`, "
           "`token_econ_report`. Community-aware retrieval (4 of the 9 "
           "tools) is the headline feature.",
    "dedup": "Semantic dedup after embed: collapse near-duplicate chunks while "
             "respecting full-doc coverage invariant.",
    "decay": "Sprint-8 relevance decay + supersession. `relevance(t) = "
             "max(backlinks, 1) * exp(-ln2 * Δt / half_life_days)`.",
    "token_econ": "Token-economy instrumentation. Per-tool cost / yield log + "
                  "report generator.",
    "scripts/bench": "Extraction-benchmark scripts (key-facts scoring, math "
                     "recall, ground-truth scaffolding).",
    "scripts/ops": "Operator scripts: stage launcher (`launch-stage.sh`), this "
                   "summary generator.",
}


def main() -> int:
    inv_path = Path("pipeline_output/codebase_inventory.jsonl")
    if not inv_path.is_file():
        print(f"ERROR: {inv_path} not present", file=sys.stderr)
        return 2
    entries = [json.loads(line) for line in inv_path.read_text().splitlines() if line.strip()]

    by_pkg: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        pkg = _bucket(e)
        if pkg is not None:
            by_pkg[pkg].append(e)

    pkg_order = [
        "_top", "ingest", "embed", "graph", "clustering", "render",
        "schema", "corpus", "retrieve", "mcp",
        "dedup", "decay", "token_econ",
        "scripts/bench", "scripts/ops",
    ]

    out: list[str] = []
    out.append("---")
    out.append('title: "Nuthatch per-script summary"')
    out.append("---")
    out.append("")
    out.append("Generated from `pipeline_output/codebase_inventory.jsonl`. "
               "Pairs with `nuthatch_pipeline.html` (Mermaid) and "
               "`nuthatch_pipeline_d2.html` (D2) for the rendered diagrams. "
               "Run `scripts/ops/generate_module_summary.py` to refresh.")
    out.append("")
    out.append("## Module graph (D2 rendering)")
    out.append("")
    out.append("![Nuthatch module graph](nuthatch_module_graph.svg)")
    out.append("")
    out.append("Source: `nuthatch_module_graph.d2`. Edges colored by "
               "source module (CLI=apricot, ingest=rust, embed=sage, "
               "graph=cream, clustering=ochre, render/MCP=dim sage, "
               "decay=brick).")
    out.append("")

    # Top-level package overview.
    out.append("## Package overview")
    out.append("")
    out.append("| Package | Files | Purpose |")
    out.append("| --- | ---: | --- |")
    for pkg in pkg_order:
        files = by_pkg.get(pkg, [])
        blurb = _PACKAGE_BLURBS.get(pkg, "_no blurb_")
        label = "_top-level" if pkg == "_top" else pkg
        out.append(f"| `{label}` | {len(files)} | {blurb} |")
    out.append("")

    # Per-package detail tables.
    for pkg in pkg_order:
        files = sorted(by_pkg.get(pkg, []), key=lambda e: e["path"])
        if not files:
            continue
        label = "Top-level" if pkg == "_top" else f"`{pkg}`"
        out.append(f"## {label}")
        out.append("")
        out.append(_PACKAGE_BLURBS.get(pkg, ""))
        out.append("")
        out.append("| Path | Purpose | Key symbols |")
        out.append("| --- | --- | --- |")
        for e in files:
            purpose = _first_paragraph(e.get("module_docstring"))
            syms = []
            for cls in e.get("classes", [])[:3]:
                syms.append(f"`{cls['name']}`")
            for fn in e.get("functions", [])[:3]:
                if fn.get("name", "").startswith("_"):
                    continue
                syms.append(f"`{fn['name']}`")
            sym_str = ", ".join(syms[:5]) or "_module-level only_"
            out.append(f"| `{e['path']}` | {purpose} | {sym_str} |")
        out.append("")

    sys.stdout.write("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
