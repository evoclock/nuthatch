#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Interactive D3.js + Canvas visualisation of a nuthatch corpus graph.

Replaces the pyvis topology view for large graphs. The performance
trick is to do all the heavy work server-side:

  1. NetworkX `forceatlas2_layout` computes node coordinates once.
  2. Communities (from a chosen `communities_<backend>.json`) supply
     per-node colours.
  3. Everything is shipped to the browser as a single JSON blob.

The browser-side is pure D3 v7 + HTML5 canvas:

  - `d3.zoom()` for pan + zoom (transform applied to the canvas
    context, not to individual elements)
  - canvas drawing in a single draw() call per zoom/pan event
  - no per-node DOM nodes, no client-side physics simulation
  - in-browser filters by entity type + relation type
  - safe DOM construction (textContent, no innerHTML on untrusted data)

Result: instant load even for 2k nodes + 34k edges, vs pyvis's
multi-minute simulation. The point of building it from scratch in
D3 is also pedagogical - to see the canvas-based rendering
primitive without any framework abstraction.

Usage:
    python scripts/viz/graph_topology_d3.py --corpus inputs
    python scripts/viz/graph_topology_d3.py --corpus inputs \\
        --communities sbm

Outputs: pipeline_output/graph_topology_d3_<backend>_<UTC>.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nuthatch.util import resolve_corpus, utc_tag
from nuthatch.util.palette import RELATION_COLORS, TYPE_COLORS


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0],
    )
    p.add_argument("--corpus", required=True, help="corpus name or path")
    p.add_argument(
        "--communities",
        default=None,
        help="suffix of the communities file to colour nodes by "
        "(e.g. 'sbm' loads communities_sbm.json). Pass 'all' "
        "to render one HTML per available communities_*.json "
        "side-by-side, sharing the layout pass.",
    )
    p.add_argument("--filter-types", default=None, help="comma-separated entity types to keep")
    p.add_argument("--max-nodes", type=int, default=3000)
    p.add_argument(
        "--layout", default="forceatlas2", choices=("forceatlas2", "spring", "kamada_kawai")
    )
    p.add_argument("--layout-iterations", type=int, default=200)
    p.add_argument("--out-dir", default="pipeline_output")
    args = p.parse_args(argv)

    from nuthatch.corpus.layout import CorpusLayout
    from nuthatch.graph.io import load_graph

    corpus_root = resolve_corpus(args.corpus)
    layout_dirs = CorpusLayout(root=corpus_root)
    print(f"[d3-viz] corpus: {layout_dirs.root}")

    graph_path = layout_dirs.kg / "graph" / "graph.json"
    if not graph_path.exists():
        print(f"[d3-viz] no graph at {graph_path}; run `nuthatch graph` first.")
        return 2
    print(f"[d3-viz] loading graph: {graph_path}")
    g = load_graph(graph_path)
    print(f"[d3-viz] {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    if args.filter_types:
        keep_types = {t.strip().lower() for t in args.filter_types.split(",")}
        keep_nodes = [
            n
            for n, d in g.nodes(data=True)
            if (d.get("entity_type") or d.get("node_type", "")).lower() in keep_types
        ]
        g = g.subgraph(keep_nodes).copy()
        print(f"[d3-viz] after type-filter: {g.number_of_nodes()} nodes")

    if g.number_of_nodes() > args.max_nodes:
        degrees = sorted(g.degree(), key=lambda kv: kv[1], reverse=True)
        keep = {n for n, _ in degrees[: args.max_nodes]}
        g = g.subgraph(keep).copy()
        print(
            f"[d3-viz] pruned to top-{args.max_nodes}: "
            f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges"
        )

    # Layout is the expensive step (~30s for 3k nodes); compute once,
    # reuse across every community overlay so --communities all is
    # cheap-per-extra-render.
    print(f"[d3-viz] computing {args.layout} layout (slow step, shared across overlays)...")
    coords = _compute_layout(g, args.layout, args.layout_iterations)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = utc_tag()

    # Build the list of (suffix, label) pairs we will render. None for
    # the no-overlay case; an empty discovery list under --communities
    # all also degrades to a single no-overlay render so the command
    # always produces at least one HTML.
    if args.communities == "all":
        overlays = _discover_overlay_suffixes(layout_dirs)
        if not overlays:
            print(
                "[d3-viz] --communities all: no communities_*.json found; "
                "rendering single no-overlay HTML."
            )
            overlays = [None]
    elif args.communities:
        overlays = [args.communities]
    else:
        overlays = [None]

    written: list[Path] = []
    for overlay in overlays:
        (
            community_by_node,
            hierarchy_by_node,
            community_labels_raw,
            comm_suffix,
        ) = _load_communities(layout_dirs, overlay)
        if community_by_node:
            n_comms = len(set(community_by_node.values()))
            n_levels = max(
                (len(chain) for chain in hierarchy_by_node.values()),
                default=1,
            )
            print(
                f"[d3-viz] overlay {comm_suffix}: {n_comms} leaf-level "
                f"communities, {n_levels} hierarchy level(s)"
            )
        elif overlay is not None:
            print(f"[d3-viz] overlay {overlay!r}: no communities file; skipping.")
            continue

        payload = _build_payload(
            g,
            coords,
            community_by_node=community_by_node,
            hierarchy_by_node=hierarchy_by_node,
            community_labels_raw=community_labels_raw,
        )
        label = comm_suffix or "nocomm"
        out_path = out_dir / f"graph_topology_d3_{label}_{tag}.html"
        out_path.write_text(
            _HTML_TEMPLATE.replace(
                "__PAYLOAD__",
                json.dumps(payload),
            )
            .replace(
                "__TITLE__",
                f"nuthatch graph ({label})",
            )
            .replace(
                "__ICON_B64__",
                _load_icon_base64(),
            ),
            encoding="utf-8",
        )
        print(f"[d3-viz] wrote: {out_path}")
        written.append(out_path)

    if not written:
        print("[d3-viz] no HTML written.")
        return 3
    return 0


def _load_icon_base64() -> str:
    """Return the nuthatch icon as a `data:image/png;base64,...` URI.

    The icon ships under `assets/Nuthatch_bgrm.png` in the tool repo.
    Embedding it directly into the HTML means the published viz has
    no external image dependency (works under air-gap and from a
    file:// URL). Returns an empty string if the asset is missing so
    the viz still renders without the icon.
    """
    import base64

    # Walk up from this module to the repo root, then into assets/.
    asset_path = Path(__file__).resolve().parents[3] / "assets" / "Nuthatch_bgrm.png"
    if not asset_path.is_file():
        return ""
    data = asset_path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _discover_overlay_suffixes(layout) -> list[str]:
    """Return all `_<suffix>` parts found in `communities_*.json` files.

    Used by `--communities all` to render one HTML per backend output
    in a single command. Sorted for deterministic order.
    """
    suffixes: list[str] = []
    if not layout.kg.is_dir():
        return suffixes
    for path in sorted(layout.kg.glob("communities_*.json")):
        stem = path.stem  # communities_sbm
        if stem.startswith("communities_"):
            suffixes.append(stem.removeprefix("communities_"))
    return suffixes


def _load_communities(
    layout,
    suffix: str | None,
) -> tuple[dict[str, int], dict[str, list[int]], dict[int, str], str]:
    """Load community membership + labels for the chosen backend's output.

    Returns `(flat_by_node, hierarchy_by_node, labels, suffix_label)`:
      - flat_by_node maps `doc::<bare_id>` -> leaf community id (int)
      - hierarchy_by_node maps `doc::<bare_id>` -> [level0, level1, ...]
        chain. For flat backends (Leiden, embeddings) this is the
        single-element list `[leaf_cid]`; for nested SBM it carries the
        full chain so the viz can offer a level selector.
      - labels maps leaf-level cid -> raw label string (either a real
        title or a `doc::<id>` placeholder; `_build_payload` resolves
        placeholders to titles using graph node data).
      - suffix_label is the backend label ('sbm', 'leiden', ...).
    """
    fname = f"communities_{suffix}.json" if suffix else "communities.json"
    path = layout.kg / fname
    if not path.exists():
        return {}, {}, {}, ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    flat = payload.get("flat") or {}
    hierarchy = payload.get("hierarchy") or {}
    raw_labels = payload.get("labels") or {}

    # Persisted keys may be bare doc_ids OR already prefixed with
    # `doc::`. Older snapshots used the bare form; current `persist.py`
    # writes prefixed. Handle both so the viz works against either.
    def _to_node_key(s: str) -> str:
        return s if s.startswith("doc::") else f"doc::{s}"

    flat_by_node: dict[str, int] = {}
    for bare, cid in flat.items():
        flat_by_node[_to_node_key(bare)] = int(cid)

    hier_by_node: dict[str, list[int]] = {}
    for bare, chain in hierarchy.items():
        key = _to_node_key(bare)
        if isinstance(chain, list) and chain:
            hier_by_node[key] = [int(c) for c in chain]
        else:
            # Fall back to single-level when hierarchy is missing
            # for this doc (defensive; persist.py should always write it).
            hier_by_node[key] = [flat_by_node.get(key, 0)]

    labels: dict[int, str] = {}
    for cid_str, raw in raw_labels.items():
        try:
            labels[int(cid_str)] = str(raw)
        except (TypeError, ValueError):
            continue

    return flat_by_node, hier_by_node, labels, (suffix or "default")


def _compute_layout(g, algorithm: str, iterations: int) -> dict:
    import networkx as nx

    if algorithm == "forceatlas2" and hasattr(nx, "forceatlas2_layout"):
        # networkx's forceatlas2 kwargs vary across versions: 3.6 dropped
        # `dissuade_hubs`. Try the keyword-augmented call first for newer
        # versions that support it, falling back to a minimal call.
        try:
            pos = nx.forceatlas2_layout(g, max_iter=iterations)
            return {str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
        except Exception as exc:
            print(f"[d3-viz] forceatlas2 failed ({exc!s}); falling back to spring_layout")

    if algorithm == "kamada_kawai":
        pos = nx.kamada_kawai_layout(g)
    else:
        pos = nx.spring_layout(g, iterations=iterations, seed=42)
    return {str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}


def _build_payload(
    g,
    coords,
    *,
    community_by_node: dict[str, int] | None = None,
    hierarchy_by_node: dict[str, list[int]] | None = None,
    community_labels_raw: dict[int, str] | None = None,
) -> dict:
    """Build the JSON payload the browser-side D3 code consumes.

    Per-node fields:
      - `type_color`: entity-type colour (used when no community overlay
        is active, or as a fallback for nodes outside the partition)
      - `community`: leaf community id (None for nodes outside the
        partition, e.g. entity nodes when the cluster only partitioned
        documents)
      - `hierarchy`: full [level0, level1, ...] chain for the level
        selector. Single-element for flat backends.

    Top-level community fields:
      - `community_labels`: cid -> human-readable name. Placeholder
        labels of the form `doc::<id>` are resolved to the referenced
        document's title (or its bare id if the title is missing).
      - `community_counts`: cid -> int member count, for legend chips.

    The browser picks community fill colour from a Catppuccin Mocha
    palette indexed by community id; the type color is the fallback.
    """
    community_by_node = community_by_node or {}
    hierarchy_by_node = hierarchy_by_node or {}
    community_labels_raw = community_labels_raw or {}

    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def _norm(v, lo, hi):
        return (v - lo) / (hi - lo) * 2 - 1 if hi > lo else 0.0

    nodes_out = []
    n_levels_observed = 0
    for node, data in g.nodes(data=True):
        x, y = coords.get(str(node), (0.0, 0.0))
        ntype = (data.get("entity_type") or data.get("node_type") or "unknown").lower()
        label = str(data.get("title") or data.get("name") or node)[:80]
        node_id = str(node)
        hierarchy = hierarchy_by_node.get(node_id)
        if hierarchy:
            n_levels_observed = max(n_levels_observed, len(hierarchy))
        nodes_out.append(
            {
                "id": node_id,
                "x": _norm(x, xmin, xmax),
                "y": _norm(y, ymin, ymax),
                "type": ntype,
                "type_color": TYPE_COLORS.get(ntype, TYPE_COLORS["unknown"]),
                "label": label,
                "degree": int(g.degree(node)),
                "community": community_by_node.get(node_id),
                "hierarchy": hierarchy,
                "summary": data.get("summary"),
            }
        )

    edges_out = []
    for u, v, data in g.edges(data=True):
        relation = data.get("relation", "related")
        edges_out.append(
            {
                "source": str(u),
                "target": str(v),
                "relation": relation,
                "color": RELATION_COLORS.get(relation, "rgba(255,255,255,0.05)"),
            }
        )

    type_counts: dict[str, int] = {}
    for n in nodes_out:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1

    relation_counts: dict[str, int] = {}
    for e in edges_out:
        relation_counts[e["relation"]] = (
            relation_counts.get(
                e["relation"],
                0,
            )
            + 1
        )

    community_counts: dict[int, int] = {}
    for n in nodes_out:
        cid = n["community"]
        if cid is None:
            continue
        community_counts[int(cid)] = community_counts.get(int(cid), 0) + 1

    # Resolve placeholder labels (doc::<id>) to the referenced
    # document's title via graph node lookup. Communities with no
    # registered label fall back to "Community <cid>".
    community_labels: dict[int, str] = {}
    for cid in community_counts:
        raw = community_labels_raw.get(cid)
        if isinstance(raw, str) and raw.startswith("doc::"):
            node_data = g.nodes.get(raw, {})
            title = node_data.get("title") or raw[len("doc::") :]
            community_labels[cid] = str(title)
        elif raw:
            community_labels[cid] = str(raw)
        else:
            community_labels[cid] = f"Community {cid}"

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "type_counts": type_counts,
        "relation_counts": relation_counts,
        "type_colors": TYPE_COLORS,
        "relation_colors": RELATION_COLORS,
        # Community-id -> count + label, for the top-bar chip legend.
        "community_counts": {str(k): v for k, v in community_counts.items()},
        "community_labels": {str(k): v for k, v in community_labels.items()},
        # n_levels = 0 means "no community overlay loaded" (HTML
        # renders by entity-type colour, hides level selector + chips).
        # n_levels >= 1 means "community overlay loaded"; the HTML
        # uses Catppuccin-by-community colour and exposes the level
        # selector when n_levels > 1 (nested SBM).
        "n_levels": int(n_levels_observed),
    }


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  /* Power Station chrome (thermall) + Catppuccin Mocha community fills.
     Layout: fixed top bar across full width, canvas fills the remaining
     viewport. Node-info card is a floating overlay (bottom-left). */
  html, body {
    margin: 0; padding: 0; background: #1a1816; color: #ead1b5;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    overflow: hidden;
  }
  #topbar {
    position: fixed; top: 0; left: 0; right: 0; height: 52px; z-index: 20;
    display: flex; align-items: center; gap: 12px;
    padding: 0 14px;
    background: #383431; border-bottom: 1px solid #5c564f;
    overflow: hidden;
  }
  #search-wrap { position: relative; flex: 0 0 220px; }
  #search {
    width: 100%; box-sizing: border-box;
    background: #1a1816; border: 1px solid #5c564f; color: #ead1b5;
    padding: 6px 10px; border-radius: 4px; font-size: 13px; outline: none;
  }
  #search:focus { border-color: #e77843; }
  #search-results {
    position: absolute; top: calc(100% + 4px); left: 0; right: 0;
    background: #1a1816; border: 1px solid #5c564f; border-radius: 4px;
    max-height: 260px; overflow-y: auto; z-index: 30; display: none;
  }
  .search-item {
    padding: 5px 8px; cursor: pointer; font-size: 12px;
    border-left: 3px solid transparent;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .search-item:hover { background: #4a4540; }

  .section { display: flex; align-items: center; gap: 6px;
             padding: 0 8px; height: 100%;
             border-left: 1px solid #4a4540; }
  .section-label {
    font-size: 10px; color: #79c39e; text-transform: uppercase;
    letter-spacing: 0.06em; padding-right: 4px;
  }
  .chips {
    display: flex; gap: 4px; overflow-x: auto; max-width: 100%;
    scrollbar-width: thin; scrollbar-color: #5c564f #383431;
  }
  .chips::-webkit-scrollbar { height: 6px; }
  .chips::-webkit-scrollbar-thumb { background: #5c564f; border-radius: 3px; }
  .chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: #4a4540; color: #ead1b5;
    border: 1px solid #5c564f; border-radius: 12px;
    padding: 3px 9px; font-size: 11px; cursor: pointer;
    white-space: nowrap; user-select: none;
    transition: opacity 0.15s, background 0.15s;
  }
  .chip:hover { background: #5c564f; }
  .chip.dimmed { opacity: 0.35; }
  .chip-dot {
    width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
  }
  .chip-count { color: #b09080; font-size: 10px; }
  #stats { margin-left: auto; font-size: 11px; color: #b09080;
           white-space: nowrap; }

  #canvas-wrap { position: fixed; top: 52px; left: 0; right: 0; bottom: 0; }
  canvas { display: block; cursor: grab; }
  canvas:active { cursor: grabbing; }

  /* Floating boxes overlaid on the canvas. Draggable via the header;
     positions persist to localStorage so they survive reloads. */
  .floating-box {
    position: fixed; z-index: 16;
    background: rgba(56, 52, 49, 0.95);
    border: 1px solid #5c564f; border-radius: 6px;
    min-width: 220px; max-width: 320px;
    max-height: calc(100vh - 120px);
    color: #ead1b5; font-size: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(6px);
    display: flex; flex-direction: column;
  }
  .floating-box-header {
    padding: 6px 10px; cursor: grab;
    background: #4a4540; border-bottom: 1px solid #5c564f;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: #79c39e; user-select: none;
    display: flex; align-items: center; justify-content: space-between;
  }
  .floating-box-header:active { cursor: grabbing; }
  .floating-box-collapse {
    background: transparent; border: none; color: #b09080;
    cursor: pointer; font-size: 14px; padding: 0 4px; line-height: 1;
  }
  .floating-box-collapse:hover { color: #ead1b5; }
  .floating-box-body {
    padding: 8px 10px; overflow-y: auto;
    scrollbar-width: thin; scrollbar-color: #5c564f #383431;
  }
  .floating-box.collapsed .floating-box-body { display: none; }
  .floating-box-subsection {
    border-top: 1px solid #4a4540; padding-top: 6px; margin-top: 6px;
  }
  .floating-box-subsection:first-child { border-top: none; padding-top: 0; margin-top: 0; }
  .floating-box-subhead {
    font-size: 10px; color: #79c39e; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 4px;
  }
  /* Chips inside floating boxes wrap rather than scroll horizontally. */
  .floating-box .chips {
    display: flex; flex-wrap: wrap; gap: 4px; overflow: visible;
    max-width: 100%;
  }

  /* Bottom-right attribution. Sits above the canvas; not draggable. */
  #attribution {
    position: fixed; bottom: 8px; right: 12px; z-index: 14;
    display: flex; align-items: center; gap: 8px;
    font-size: 10px; color: #b09080;
    background: rgba(26, 24, 22, 0.55);
    padding: 4px 8px; border-radius: 4px;
    backdrop-filter: blur(4px);
  }
  #attribution img {
    width: 18px; height: 18px;
    image-rendering: pixelated; image-rendering: -moz-crisp-edges;
  }
  #attribution a { color: #79c39e; text-decoration: none; }
  #attribution a:hover { text-decoration: underline; }

  #info {
    position: fixed; bottom: 14px; left: 14px;
    background: rgba(26, 24, 22, 0.95);
    border: 1px solid #79c39e; border-radius: 6px;
    padding: 12px 14px; font-size: 12px; color: #ead1b5;
    backdrop-filter: blur(4px); z-index: 15;
    display: none; max-width: 360px; max-height: 320px; overflow-y: auto;
  }
  #info h4 { margin: 0 0 4px 0; font-size: 14px; color: #ead1b5; }
  #info .field { font-size: 11px; color: #b09080; margin-bottom: 2px; }
  #info .field b { color: #ead1b5; font-weight: 600; }
  #info .neighbors-header {
    margin-top: 8px; color: #79c39e; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  #info .neighbor {
    display: block; padding: 2px 6px; margin: 2px 0;
    border-radius: 3px; cursor: pointer; font-size: 11px;
    border-left: 3px solid #5c564f;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  #info .neighbor:hover { background: #4a4540; }

  code { color: #79c39e; font-family: ui-monospace, monospace; }
</style>
</head>
<body>
<div id="topbar">
  <div id="search-wrap">
    <input id="search" type="text" placeholder="Search nodes..."
           autocomplete="off">
    <div id="search-results"></div>
  </div>
  <div id="level-section" class="section" style="display:none">
    <span class="section-label">Level</span>
    <div id="level-select-wrap"></div>
  </div>
  <div id="stats"></div>
</div>

<div id="canvas-wrap"><canvas id="c"></canvas></div>

<!-- Floating box 1: Communities chips. Hidden when no overlay loaded. -->
<div id="box-communities" class="floating-box" style="display:none">
  <div class="floating-box-header" data-box-id="communities">
    <span>Communities</span>
    <button class="floating-box-collapse" type="button"
            aria-label="Toggle">&minus;</button>
  </div>
  <div class="floating-box-body">
    <div id="community-chips" class="chips"></div>
  </div>
</div>

<!-- Floating box 2: Types + Relations folded together. -->
<div id="box-filters" class="floating-box">
  <div class="floating-box-header" data-box-id="filters">
    <span>Filters</span>
    <button class="floating-box-collapse" type="button"
            aria-label="Toggle">&minus;</button>
  </div>
  <div class="floating-box-body">
    <div class="floating-box-subsection">
      <div class="floating-box-subhead">Types</div>
      <div id="type-chips" class="chips"></div>
    </div>
    <div class="floating-box-subsection">
      <div class="floating-box-subhead">Relations</div>
      <div id="relation-chips" class="chips"></div>
    </div>
  </div>
</div>

<div id="info"></div>

<div id="attribution">
  <img src="__ICON_B64__" alt="nuthatch" />
  <span>
    <a href="https://github.com/" target="_blank" rel="noopener">nuthatch</a>
    &middot; Apache-2.0 &middot; &copy; 2026 Julen Gamboa
  </span>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
"use strict";
const PAYLOAD = __PAYLOAD__;

// ---- safe DOM helpers ------------------------------------------------
function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "style" && typeof v === "object") {
        for (const [sk, sv] of Object.entries(v)) e.style[sk] = sv;
      } else if (k.startsWith("on") && typeof v === "function") {
        e.addEventListener(k.slice(2), v);
      } else if (k === "for") {
        e.htmlFor = v;
      } else {
        e.setAttribute(k, v);
      }
    }
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

// ---- Canvas + DPR-aware sizing ---------------------------------------
// Canvas lives under a 52px top bar; sizing accounts for that so the
// projected coordinates and hit-tests stay aligned.
const TOPBAR_H = 52;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const dpr = window.devicePixelRatio || 1;
function viewportW() { return window.innerWidth; }
function viewportH() { return Math.max(0, window.innerHeight - TOPBAR_H); }
function resize() {
  canvas.width = viewportW() * dpr;
  canvas.height = viewportH() * dpr;
  canvas.style.width = viewportW() + "px";
  canvas.style.height = viewportH() + "px";
  draw();
}
window.addEventListener("resize", resize);

// ---- Community colouring (Catppuccin Mocha named accents) ----------
// Ordered so adjacent community ids land on contrasting hues. Wraps
// with HSL-rotated variants past the named palette length so any
// number of communities gets a stable, distinguishable colour.
// Must stay byte-identical to `_CATPPUCCIN_HEX` in
// `render/obsidian.py` so a community renders the same colour in
// Obsidian and in this D3 viz.
const CATPPUCCIN = [
  "#89B4FA", // blue
  "#FAB387", // peach
  "#A6E3A1", // green
  "#F38BA8", // pink
  "#94E2D5", // teal
  "#F9E2AF", // yellow
  "#CBA6F7", // mauve
  "#EBA0AC", // maroon
  "#74C7EC", // sapphire
  "#F5C2E7", // flamingo
  "#179299", // Latte teal (replaces lavender)
  "#89DCEB", // sky
  "#F5E0DC", // rosewater
  "#A6ADC8", // subtext1
];
function communityColor(cid) {
  if (cid == null) return null;
  const i = ((cid % CATPPUCCIN.length) + CATPPUCCIN.length) %
            CATPPUCCIN.length;
  return CATPPUCCIN[i];
}

// Active hierarchy level. 0 = leaf (finest), n_levels-1 = root
// (coarsest meta-community). Selector only shown when n_levels > 1.
let activeLevel = 0;

function nodeColor(n) {
  // No community overlay loaded -> entity-type colour.
  if (PAYLOAD.n_levels === 0) return n.type_color;
  // Pick community id at active level from the per-node hierarchy
  // chain. Falls back to leaf if the chain is shorter than asked.
  if (n.hierarchy && n.hierarchy.length > 0) {
    const idx = Math.min(activeLevel, n.hierarchy.length - 1);
    const cid = n.hierarchy[idx];
    const col = communityColor(cid);
    if (col) return col;
  }
  return n.type_color;
}

// ---- Selection state (click-to-highlight a node + its edges) ---------
let selectedId = null;

// Precompute neighbour set per node so the per-frame draw doesn't
// re-scan edges to decide what to dim. One pass at load time.
const neighborsOf = new Map();
for (const e of PAYLOAD.edges) {
  if (!neighborsOf.has(e.source)) neighborsOf.set(e.source, new Set());
  if (!neighborsOf.has(e.target)) neighborsOf.set(e.target, new Set());
  neighborsOf.get(e.source).add(e.target);
  neighborsOf.get(e.target).add(e.source);
}

function isHighlighted(id) {
  if (selectedId === null) return true;
  if (id === selectedId) return true;
  return neighborsOf.get(selectedId)?.has(id) ?? false;
}

// ---- Filter state ---------------------------------------------------
// Three independent dim/hide gates: by type, by relation, by community.
// `activeCommunities` is only populated when an overlay is loaded.
const activeTypes = new Set(Object.keys(PAYLOAD.type_counts));
const activeRels = new Set(Object.keys(PAYLOAD.relation_counts));
const activeCommunities = new Set(
  Object.keys(PAYLOAD.community_counts || {}).map(s => parseInt(s, 10))
);

function chip({ label, color, count, active, onclick }) {
  const c = el("span", {
    class: active ? "chip" : "chip dimmed",
    onclick: onclick,
  });
  if (color) {
    c.appendChild(el("span", { class: "chip-dot",
                                style: { background: color } }));
  }
  c.appendChild(document.createTextNode(label));
  if (count != null) {
    c.appendChild(el("span", { class: "chip-count" }, " ", String(count)));
  }
  return c;
}

function buildTypeChips() {
  const div = document.getElementById("type-chips");
  for (const [type, count] of Object.entries(PAYLOAD.type_counts)) {
    const color = PAYLOAD.type_colors[type] || "#666";
    const c = chip({
      label: type, color, count, active: true,
      onclick: () => {
        if (activeTypes.has(type)) {
          activeTypes.delete(type);
          c.classList.add("dimmed");
        } else {
          activeTypes.add(type);
          c.classList.remove("dimmed");
        }
        draw();
      },
    });
    div.appendChild(c);
  }
}

function buildRelationChips() {
  const div = document.getElementById("relation-chips");
  for (const [rel, count] of Object.entries(PAYLOAD.relation_counts)) {
    const c = chip({
      label: rel, color: null, count, active: true,
      onclick: () => {
        if (activeRels.has(rel)) {
          activeRels.delete(rel);
          c.classList.add("dimmed");
        } else {
          activeRels.add(rel);
          c.classList.remove("dimmed");
        }
        draw();
      },
    });
    div.appendChild(c);
  }
}

function buildCommunityChips() {
  if (PAYLOAD.n_levels === 0) return;
  const box = document.getElementById("box-communities");
  const div = document.getElementById("community-chips");
  box.style.display = "flex";
  const entries = Object.entries(PAYLOAD.community_counts || {})
    .map(([k, v]) => [parseInt(k, 10), v])
    .sort((a, b) => b[1] - a[1]);  // largest first
  for (const [cid, count] of entries) {
    const color = communityColor(cid);
    const label = PAYLOAD.community_labels[String(cid)] || `Community ${cid}`;
    const c = chip({
      label, color, count, active: true,
      onclick: () => {
        if (activeCommunities.has(cid)) {
          activeCommunities.delete(cid);
          c.classList.add("dimmed");
        } else {
          activeCommunities.add(cid);
          c.classList.remove("dimmed");
        }
        draw();
      },
    });
    div.appendChild(c);
  }
}

function buildLevelSelector() {
  // Only show the level selector when the loaded overlay actually has
  // hierarchy depth > 1 (i.e., SBM). Flat backends (Leiden,
  // embeddings) keep the section hidden.
  if (PAYLOAD.n_levels <= 1) return;
  document.getElementById("level-section").style.display = "flex";
  const div = document.getElementById("level-select-wrap");
  const select = el("select", {
    style: { background: "#1a1816", color: "#ead1b5",
              border: "1px solid #5c564f", borderRadius: "4px",
              padding: "3px 6px", fontSize: "11px" },
    onchange: (e) => {
      activeLevel = parseInt(e.target.value, 10);
      draw();
    },
  });
  for (let i = 0; i < PAYLOAD.n_levels; i++) {
    const opt = el("option", { value: String(i) },
      `L${i}` + (i === 0 ? " (leaf)" :
                  i === PAYLOAD.n_levels - 1 ? " (root)" : ""),
    );
    if (i === 0) opt.selected = true;
    select.appendChild(opt);
  }
  div.appendChild(select);
}

function buildStats() {
  const n = PAYLOAD.nodes.length;
  const e = PAYLOAD.edges.length;
  const k = Object.keys(PAYLOAD.community_counts || {}).length;
  const parts = [`${n} nodes`, `${e} edges`];
  if (k > 0) parts.push(`${k} communities`);
  document.getElementById("stats").textContent = parts.join("  ·  ");
}

// ---- Floating-box drag + collapse + localStorage persistence --------
// Each `.floating-box` has a `.floating-box-header` with a
// `data-box-id` attribute used as the localStorage key. Position +
// collapsed state are persisted; on load, restored if present and
// still within the current viewport bounds (defended against viewport
// shrinkage that would otherwise leave a box off-screen).
function _floatingBoxStorageKey(id, field) {
  return `nuthatch-d3-viz.box.${id}.${field}`;
}
function _clampToViewport(x, y, box) {
  const rect = box.getBoundingClientRect();
  const w = rect.width || 240;
  const h = rect.height || 80;
  const maxX = Math.max(0, window.innerWidth - w - 8);
  const maxY = Math.max(0, window.innerHeight - h - 8);
  // Keep top-left at least 4px in from each edge.
  return [
    Math.min(Math.max(4, x), maxX),
    Math.min(Math.max(TOPBAR_H + 4, y), maxY),
  ];
}
function _applyStoredFloatingState(box) {
  const header = box.querySelector(".floating-box-header");
  const id = header.dataset.boxId;
  const xRaw = localStorage.getItem(_floatingBoxStorageKey(id, "x"));
  const yRaw = localStorage.getItem(_floatingBoxStorageKey(id, "y"));
  const collapsedRaw = localStorage.getItem(
    _floatingBoxStorageKey(id, "collapsed"),
  );
  if (xRaw != null && yRaw != null) {
    const [x, y] = _clampToViewport(
      parseInt(xRaw, 10), parseInt(yRaw, 10), box,
    );
    box.style.left = x + "px";
    box.style.top = y + "px";
    // Clear the right/bottom defaults so left/top win.
    box.style.right = "auto";
    box.style.bottom = "auto";
  }
  if (collapsedRaw === "1") {
    box.classList.add("collapsed");
    const btn = box.querySelector(".floating-box-collapse");
    if (btn) btn.textContent = "+";
  }
}
function _wireFloatingBox(box) {
  const header = box.querySelector(".floating-box-header");
  const collapseBtn = box.querySelector(".floating-box-collapse");
  const id = header.dataset.boxId;

  // Drag.
  let dragging = false;
  let dx = 0; let dy = 0;
  header.addEventListener("mousedown", (e) => {
    // Ignore drags that start on the collapse button.
    if (e.target.classList.contains("floating-box-collapse")) return;
    dragging = true;
    const rect = box.getBoundingClientRect();
    dx = e.clientX - rect.left;
    dy = e.clientY - rect.top;
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const [x, y] = _clampToViewport(e.clientX - dx, e.clientY - dy, box);
    box.style.left = x + "px";
    box.style.top = y + "px";
    box.style.right = "auto";
    box.style.bottom = "auto";
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    const rect = box.getBoundingClientRect();
    localStorage.setItem(
      _floatingBoxStorageKey(id, "x"), String(Math.round(rect.left)),
    );
    localStorage.setItem(
      _floatingBoxStorageKey(id, "y"), String(Math.round(rect.top)),
    );
  });

  // Collapse toggle.
  collapseBtn.addEventListener("click", () => {
    box.classList.toggle("collapsed");
    const collapsed = box.classList.contains("collapsed");
    collapseBtn.textContent = collapsed ? "+" : "-";
    localStorage.setItem(
      _floatingBoxStorageKey(id, "collapsed"), collapsed ? "1" : "0",
    );
  });
}
function initFloatingBoxes() {
  // Default placement: top-right corner, stacked. The box-communities
  // box sits above box-filters when both are visible.
  const right = 14;
  const commBox = document.getElementById("box-communities");
  const filtBox = document.getElementById("box-filters");
  if (commBox) {
    commBox.style.top = (TOPBAR_H + 14) + "px";
    commBox.style.right = right + "px";
    _wireFloatingBox(commBox);
    _applyStoredFloatingState(commBox);
  }
  if (filtBox) {
    // Place filters below communities when both visible; otherwise
    // top-right. The localStorage restore overrides this default.
    filtBox.style.top = (commBox && commBox.style.display !== "none"
                          ? TOPBAR_H + 14 + 280 : TOPBAR_H + 14) + "px";
    filtBox.style.right = right + "px";
    _wireFloatingBox(filtBox);
    _applyStoredFloatingState(filtBox);
  }
}

buildTypeChips();
buildRelationChips();
buildCommunityChips();
buildLevelSelector();
buildStats();
initFloatingBoxes();

// ---- Zoom / pan via d3-zoom ------------------------------------------
let transform = d3.zoomIdentity;
d3.select(canvas).call(
  d3.zoom().scaleExtent([0.1, 12])
    .on("zoom", (e) => { transform = e.transform; draw(); })
);

// ---- Coordinate mapping ----------------------------------------------
function project(x, y) {
  const m = 50;
  const W = viewportW() - 2 * m;
  const H = viewportH() - 2 * m;
  return [m + (x + 1) * 0.5 * W, m + (y + 1) * 0.5 * H];
}

// ---- Index for hover hit-testing -------------------------------------
const nodeIndex = new Map();
for (const n of PAYLOAD.nodes) nodeIndex.set(n.id, n);

// ---- Draw loop -------------------------------------------------------
function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#1a1816";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.scale(dpr, dpr);
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.k, transform.k);

  // Community filter is inert when no overlay is loaded (the set is
  // empty in that case, and nodeVisible() short-circuits on null cid).
  function nodeVisible(n) {
    if (!activeTypes.has(n.type)) return false;
    if (PAYLOAD.n_levels > 0 && n.community != null &&
        !activeCommunities.has(n.community)) return false;
    return true;
  }

  ctx.lineWidth = 0.6 / transform.k;
  for (const e of PAYLOAD.edges) {
    if (!activeRels.has(e.relation)) continue;
    const s = nodeIndex.get(e.source);
    const t = nodeIndex.get(e.target);
    if (!s || !t) continue;
    if (!nodeVisible(s) || !nodeVisible(t)) continue;
    // Selection: when a node is selected, only edges touching it draw
    // at full opacity; other edges fade so the local neighbourhood
    // pops. When no selection is active, every edge uses its base
    // colour (current behaviour).
    const edgeIsLive = selectedId === null ||
                       e.source === selectedId || e.target === selectedId;
    const [x1, y1] = project(s.x, s.y);
    const [x2, y2] = project(t.x, t.y);
    ctx.strokeStyle = edgeIsLive ? e.color : "rgba(255,255,255,0.015)";
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  for (const n of PAYLOAD.nodes) {
    if (!nodeVisible(n)) continue;
    const [x, y] = project(n.x, n.y);
    const r = Math.max(2, Math.min(8, 2 + Math.log2(n.degree + 1))) /
              Math.sqrt(transform.k);
    // Selection-aware fill: highlighted nodes use their base colour
    // (entity-type OR community), dimmed nodes use neutral grey at
    // low alpha so the selected neighbourhood pops.
    if (isHighlighted(n.id)) {
      ctx.fillStyle = nodeColor(n);
    } else {
      ctx.fillStyle = "rgba(80,80,80,0.35)";
    }
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    // White ring on the actively selected node.
    if (n.id === selectedId) {
      ctx.lineWidth = Math.max(1.5 / transform.k, 1);
      ctx.strokeStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x, y, r + 3 / transform.k, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

// ---- Node info card (click-driven, safe DOM) -------------------------
const info = document.getElementById("info");

function hitTest(event) {
  const rect = canvas.getBoundingClientRect();
  const mx = (event.clientX - rect.left - transform.x) / transform.k;
  const my = (event.clientY - rect.top  - transform.y) / transform.k;
  let best = null;
  let bestD = 14;
  for (const n of PAYLOAD.nodes) {
    if (!activeTypes.has(n.type)) continue;
    const [x, y] = project(n.x, n.y);
    const d = Math.hypot(x - mx, y - my);
    if (d < bestD) { bestD = d; best = n; }
  }
  return best;
}

function showInfo(node) {
  while (info.firstChild) info.removeChild(info.firstChild);
  info.appendChild(el("h4", null, node.label));
  info.appendChild(el("div", { class: "field" },
    el("b", null, "id: "), el("code", null, node.id)));
  info.appendChild(el("div", { class: "field" },
    el("b", null, "type: "), node.type,
    "  ·  ",
    el("b", null, "degree: "), String(node.degree)));
  if (node.community != null) {
    const label = PAYLOAD.community_labels[String(node.community)] ||
                  `Community ${node.community}`;
    info.appendChild(el("div", { class: "field" },
      el("b", null, "community: "), label,
      el("span", { style: { color: "#b09080" } }, ` (#${node.community})`)));
  }
  if (node.summary) {
    info.appendChild(el("div", { class: "field",
                                  style: { marginTop: "6px",
                                            fontStyle: "italic" } },
                       node.summary));
  }
  const neighborIds = Array.from(neighborsOf.get(node.id) || []);
  if (neighborIds.length > 0) {
    info.appendChild(el("div", { class: "neighbors-header" },
      `Neighbors (${neighborIds.length})`));
    // Cap the rendered list at 50 for very high-degree hubs; the
    // scroll affordance is on the panel itself.
    const shown = neighborIds.slice(0, 50);
    for (const nid of shown) {
      const nb = nodeIndex.get(nid);
      if (!nb) continue;
      const color = communityColor(nb.community) || nb.type_color || "#5c564f";
      const link = el("span", {
        class: "neighbor",
        style: { borderLeftColor: color },
        onclick: () => focusNode(nid),
      }, nb.label);
      info.appendChild(link);
    }
    if (neighborIds.length > shown.length) {
      info.appendChild(el("div", { class: "field",
                                    style: { marginTop: "4px" } },
        `… and ${neighborIds.length - shown.length} more`));
    }
  }
  info.style.display = "block";
}

function focusNode(nodeId) {
  const n = nodeIndex.get(nodeId);
  if (!n) return;
  selectedId = nodeId;
  // Re-centre on the node and bump scale to 2x (or current, whichever
  // is larger). Canvas coords are 0..viewport, so centering math uses
  // viewportW/H, not window dimensions.
  const [px, py] = project(n.x, n.y);
  const targetScale = Math.max(2, transform.k);
  d3.select(canvas).transition().duration(400).call(
    d3.zoom().transform,
    d3.zoomIdentity
      .translate(viewportW() / 2 - px * targetScale,
                 viewportH() / 2 - py * targetScale)
      .scale(targetScale),
  );
  showInfo(n);
}

canvas.addEventListener("click", (event) => {
  const node = hitTest(event);
  if (node) {
    selectedId = (node.id !== selectedId) ? node.id : null;
    if (selectedId) showInfo(node);
    else info.style.display = "none";
  } else {
    // Empty-canvas click clears selection + dismisses the card.
    selectedId = null;
    info.style.display = "none";
  }
  draw();
});

// ---- Search ----------------------------------------------------------
const searchInput = document.getElementById("search");
const searchResults = document.getElementById("search-results");
searchInput.addEventListener("input", () => {
  const q = searchInput.value.toLowerCase().trim();
  while (searchResults.firstChild) {
    searchResults.removeChild(searchResults.firstChild);
  }
  if (!q) { searchResults.style.display = "none"; return; }
  const matches = PAYLOAD.nodes
    .filter(n => n.label.toLowerCase().includes(q) ||
                  n.id.toLowerCase().includes(q))
    .slice(0, 20);
  if (matches.length === 0) {
    searchResults.style.display = "none";
    return;
  }
  for (const n of matches) {
    const color = communityColor(n.community) || n.type_color || "#5c564f";
    const item = el("div", {
      class: "search-item",
      style: { borderLeftColor: color },
      onclick: () => {
        focusNode(n.id);
        searchResults.style.display = "none";
        searchInput.value = "";
        draw();
      },
    }, n.label);
    searchResults.appendChild(item);
  }
  searchResults.style.display = "block";
});
document.addEventListener("click", (e) => {
  if (e.target !== searchInput && !searchResults.contains(e.target)) {
    searchResults.style.display = "none";
  }
});

resize();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
