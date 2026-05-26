# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""nuthatch command-line interface (Sprint 1 surface).

Subcommands:

    nuthatch init <path>           Scaffold a new corpus at <path>.
    nuthatch ingest [--corpus N]   One-shot: process everything in inbox/.
    nuthatch watch  [--corpus N]   Long-running: ingest on filesystem events.
    nuthatch status [--corpus N]   Print manifest stats + per-status counts.
    nuthatch corpus list           List registered corpora.

A corpus is resolved from (in priority order):
    1. `--corpus <name>` arg via the registry at
       `~/.config/nuthatch/registry.toml` (or `$XDG_CONFIG_HOME`).
    2. `--corpus <path>` if the arg looks like a path that exists.
    3. Walking up from the current directory looking for `.kg/`.
    4. Registry default (`default_corpus`).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from types import FrameType

from nuthatch import __version__
from nuthatch.corpus import (
    CorpusLayout,
    Registry,
    discover_corpus_root,
    init_corpus,
)
from nuthatch.corpus.config import load_corpus_config
from nuthatch.decay import render_report, run_decay_pass
from nuthatch.embed.orchestrator import embed_corpus
from nuthatch.ingest import IngestOrchestrator, IngestResult
from nuthatch.ingest.manifest import IngestStatus, ManifestStore
from nuthatch.ingest.watch import InboxWatcher
from nuthatch.token_econ.log import TokenLog
from nuthatch.token_econ.report import (
    aggregate,
    render_markdown_report,
    summary_as_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nuthatch",
        description=(
            "Knowledge-graph tool for paper corpora with principled clustering "
            "(Sprint 1: corpus state machine)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"nuthatch {__version__}")

    subparsers = parser.add_subparsers(dest="subcommand", required=False)

    init_p = subparsers.add_parser("init", help="scaffold a new corpus at <path>")
    init_p.add_argument("path", type=Path, help="directory to initialise as a corpus")
    init_p.add_argument(
        "--register-as",
        type=str,
        default=None,
        help="name to register the corpus under (omit to skip registry update)",
    )
    init_p.add_argument(
        "--set-default",
        action="store_true",
        help="also mark the corpus as the registry's default",
    )

    ingest_p = subparsers.add_parser("ingest", help="one-shot: process inbox/")
    _add_corpus_arg(ingest_p)

    watch_p = subparsers.add_parser("watch", help="long-running: ingest on FS events")
    _add_corpus_arg(watch_p)

    status_p = subparsers.add_parser("status", help="manifest stats for the corpus")
    _add_corpus_arg(status_p)

    corpus_p = subparsers.add_parser("corpus", help="corpus registry operations")
    corpus_sub = corpus_p.add_subparsers(dest="corpus_command", required=True)
    corpus_sub.add_parser("list", help="list registered corpora")

    report_p = subparsers.add_parser(
        "token-report",
        help="aggregate the token-economy log into stats + markdown",
    )
    _add_corpus_arg(report_p)
    report_p.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO-8601 UTC lower bound (e.g. 2026-05-01 or 2026-05-01T00:00:00Z)",
    )
    report_p.add_argument(
        "--until",
        type=str,
        default=None,
        help="ISO-8601 UTC upper bound (inclusive)",
    )
    report_p.add_argument(
        "--group-by",
        type=str,
        choices=("tool", "day", "surface"),
        default="tool",
        help="grouping axis for the per-row breakdown",
    )
    report_p.add_argument(
        "--tool",
        type=str,
        default=None,
        help="filter to one tool name",
    )
    report_p.add_argument(
        "--surface-id",
        type=str,
        default=None,
        help="filter to one surface (e.g. 'mcp', 'cli')",
    )
    report_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "write the markdown report to this path. "
            "Default: <corpus>/reports/token-economy-<YYYY-MM-DD>.md. "
            "Pass '-' to print to stdout instead of writing."
        ),
    )
    report_p.add_argument(
        "--json",
        action="store_true",
        help="print the JSON summary to stdout in addition to writing markdown",
    )

    decay_p = subparsers.add_parser(
        "decay",
        help="apply relevance decay + supersession across the corpus",
    )
    _add_corpus_arg(decay_p)
    decay_p.add_argument(
        "--dry-run",
        action="store_true",
        help="compute the pass but do not write cards or graph back to disk",
    )
    decay_p.add_argument(
        "--report-only",
        action="store_true",
        help="print only the markdown report (implies --dry-run unless --commit is set)",
    )
    decay_p.add_argument(
        "--commit",
        action="store_true",
        help="with --report-only, still write cards and graph back to disk",
    )
    decay_p.add_argument(
        "--archive-threshold",
        type=float,
        default=0.1,
        help="post-decay relevance below this lists the doc as an archive candidate",
    )
    decay_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "write the markdown report to this path. "
            "Default: <corpus>/reports/decay-<YYYY-MM-DD>.md. "
            "Pass '-' to print to stdout instead of writing."
        ),
    )

    embed_p = subparsers.add_parser(
        "embed",
        help=(
            "chunk + embed every doc in <corpus>/.kg/extracted/ into "
            "the corpus vector store. Incremental by default."
        ),
    )
    _add_corpus_arg(embed_p)
    embed_p.add_argument(
        "--force",
        action="store_true",
        help=(
            "re-embed every doc from scratch (deletes existing chunks "
            "per doc first). Required after changing _DEFAULT_MAX_CHARS / "
            "_DEFAULT_OVERLAP_CHARS in src/nuthatch/embed/chunk.py or "
            "switching the embedding model via .kg/config.yaml."
        ),
    )

    graph_p = subparsers.add_parser(
        "graph",
        help=(
            "build the corpus graph from <corpus>/.kg/extracted/. "
            "Entity extraction + node/edge assembly."
        ),
    )
    _add_corpus_arg(graph_p)

    cluster_p = subparsers.add_parser(
        "cluster",
        help=(
            "fit communities on the corpus graph. Full refit each call "
            "(SBM / Leiden / embeddings, highest-rigor available)."
        ),
    )
    _add_corpus_arg(cluster_p)

    render_p = subparsers.add_parser(
        "render",
        help=(
            "regenerate Obsidian-compatible cards/, communities/, "
            "dashboard.md, index.md from current corpus state."
        ),
    )
    _add_corpus_arg(render_p)

    return parser


def _add_corpus_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--corpus",
        type=str,
        default=None,
        help=(
            "registered corpus name OR a path to a corpus directory. "
            "If omitted, walks up from cwd looking for a .kg/ marker, "
            "then falls back to the registry's default_corpus."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    if args.subcommand == "init":
        return _cmd_init(args)
    if args.subcommand == "ingest":
        return _cmd_ingest(args)
    if args.subcommand == "watch":
        return _cmd_watch(args)
    if args.subcommand == "status":
        return _cmd_status(args)
    if args.subcommand == "corpus" and args.corpus_command == "list":
        return _cmd_corpus_list()
    if args.subcommand == "token-report":
        return _cmd_token_report(args)
    if args.subcommand == "decay":
        return _cmd_decay(args)
    if args.subcommand == "embed":
        return _cmd_embed(args)
    if args.subcommand == "graph":
        return _cmd_graph(args)
    if args.subcommand == "cluster":
        return _cmd_cluster(args)
    if args.subcommand == "render":
        return _cmd_render(args)

    parser.print_help()
    return 2


def _cmd_init(args: argparse.Namespace) -> int:
    layout = init_corpus(args.path)
    print(f"[OK] initialised corpus at {layout.root}")
    print(f"     marker:   {layout.kg}")
    print(f"     inbox:    {layout.inbox}")
    print(f"     papers:   {layout.papers}")
    print(f"     manifest: {layout.manifest_path}")

    if args.register_as is not None:
        registry = Registry.load()
        registry.add(args.register_as, layout.root)
        if args.set_default:
            registry.set_default(args.register_as)
        registry.save()
        suffix = " (default)" if args.set_default else ""
        print(f"[OK] registered as {args.register_as!r}{suffix}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    layout = _resolve_layout_or_die(args)
    orchestrator = IngestOrchestrator(layout)
    results = orchestrator.ingest_inbox()
    _print_results(results)
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    layout = _resolve_layout_or_die(args)
    orchestrator = IngestOrchestrator(layout)

    def on_pass(results: list[IngestResult]) -> None:
        if results:
            print(f"[watcher] processed {len(results)} file(s)")
            _print_results(results)

    watcher = InboxWatcher(orchestrator, on_result=on_pass)
    watcher.start()
    print(f"[watcher] watching {layout.inbox} (Ctrl-C to stop)")

    stop_event = {"stop": False}

    def _on_signal(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        stop_event["stop"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        while not stop_event["stop"]:
            time.sleep(0.5)
    finally:
        watcher.stop()
        print("[watcher] stopped")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    layout = _resolve_layout_or_die(args)
    store = ManifestStore(layout.manifest_path)
    counts: Counter[str] = Counter()
    total = 0
    for entry in store.iter_entries():
        counts[entry.status.value] += 1
        total += 1
    print(f"corpus:   {layout.root}")
    print(f"manifest: {layout.manifest_path}")
    print(f"total:    {total}")
    for status in IngestStatus:
        print(f"  {status.value:11} {counts.get(status.value, 0)}")
    return 0


def _cmd_corpus_list() -> int:
    registry = Registry.load()
    if not registry.corpora:
        print("(registry is empty; run `nuthatch init <path> --register-as <name>`)")
        return 0
    default = registry.default_corpus
    for name in registry.names():
        marker = " *" if name == default else "  "
        path = registry.resolve(name)
        print(f"{marker} {name:24} {path}")
    return 0


def _resolve_layout_or_die(args: argparse.Namespace) -> CorpusLayout:
    """Resolve a `CorpusLayout` from `--corpus` arg or directory walk-up."""
    raw = getattr(args, "corpus", None)
    registry = Registry.load()

    candidate: Path | None = None
    if raw is not None:
        as_path = Path(raw).expanduser()
        if as_path.is_dir():
            candidate = as_path
        else:
            resolved = registry.resolve(raw)
            if resolved is not None and resolved.is_dir():
                candidate = resolved
    else:
        discovered = discover_corpus_root(Path.cwd())
        if discovered is not None:
            candidate = discovered
        else:
            resolved = registry.resolve(None)
            if resolved is not None and resolved.is_dir():
                candidate = resolved

    if candidate is None:
        print(
            "error: could not resolve a corpus. Pass --corpus <name|path>, run from "
            "inside a corpus (a directory with a .kg/ marker), or register a default "
            "with `nuthatch init <path> --register-as <name> --set-default`.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return CorpusLayout(root=candidate.resolve())


def _cmd_token_report(args: argparse.Namespace) -> int:
    import json as _json
    from datetime import UTC, datetime

    layout = _resolve_layout_or_die(args)
    log_path = layout.kg / "token_log.jsonl"
    token_log = TokenLog(log_path)
    records = list(
        token_log.iter_records(
            since=args.since,
            until=args.until,
            tool=args.tool,
            surface_id=args.surface_id,
        )
    )
    summary = aggregate(records, group_by=args.group_by)

    md = render_markdown_report(summary, corpus_name=layout.root.name)

    if args.out is not None and str(args.out) == "-":
        sys.stdout.write(md)
    else:
        out_path = args.out
        if out_path is None:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")
            out_path = layout.root / "reports" / f"token-economy-{date_str}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"[OK] wrote token-economy report to {out_path}")
        print(
            f"     {summary.n_queries} queries; "
            f"{summary.tokens_served_total:,} served vs "
            f"{summary.tokens_counterfactual_total:,} counterfactual "
            f"({summary.pct_saved}% saved)."
        )

    if args.json:
        sys.stdout.write(_json.dumps(summary_as_dict(summary), indent=2) + "\n")
    return 0


def _cmd_decay(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    layout = _resolve_layout_or_die(args)
    # --report-only implies dry-run unless --commit is explicit.
    dry_run = args.dry_run or (args.report_only and not args.commit)
    result = run_decay_pass(
        layout,
        dry_run=dry_run,
        archive_threshold=args.archive_threshold,
    )
    report_md = render_report(result, corpus_name=layout.root.name)

    if args.report_only:
        if args.out is not None and str(args.out) == "-":
            sys.stdout.write(report_md)
        else:
            out_path = args.out
            if out_path is None:
                date_str = datetime.now(UTC).strftime("%Y-%m-%d")
                out_path = layout.root / "reports" / f"decay-{date_str}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report_md, encoding="utf-8")
            print(f"[OK] wrote decay report to {out_path}")
        return 0

    # Default: print a terse summary; full report still written to disk
    # unless the user redirected with --out -.
    out_path = args.out
    if out_path is None:
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        out_path = layout.root / "reports" / f"decay-{date_str}.md"
    if str(out_path) == "-":
        sys.stdout.write(report_md)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_md, encoding="utf-8")
        print(
            f"[OK] decay pass {'dry-run' if dry_run else 'committed'}: "
            f"scanned {result.n_cards_scanned}, "
            f"decayed {result.n_decayed}, "
            f"pinned {result.n_pinned}, "
            f"superseded {result.n_superseded_this_pass}, "
            f"archive candidates {len(result.archive_candidates)}"
        )
        print(f"[OK] wrote decay report to {out_path}")
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    layout = _resolve_layout_or_die(args)
    config = load_corpus_config(layout.config_path)
    result = embed_corpus(layout, config=config, force=args.force)
    print(
        f"[OK] embed: scanned {result.n_docs_scanned}, "
        f"embedded {result.n_docs_embedded}, "
        f"skipped {result.n_docs_skipped} (already in store), "
        f"chunks added {result.n_chunks_added}"
        + (" [--force]" if result.forced else "")
    )
    if result.failed:
        print(f"[WARN] {len(result.failed)} docs failed (first 5 shown):")
        for doc_id, err in result.failed[:5]:
            print(f"  - {doc_id}: {err[:200]}")
        return 1
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    from nuthatch.graph.orchestrator import build_graph_for_corpus

    layout = _resolve_layout_or_die(args)
    result = build_graph_for_corpus(layout)
    print(
        f"[OK] graph: {result.n_docs} docs -> "
        f"{result.n_nodes} nodes, {result.n_edges} edges"
    )
    print(f"     graph: {result.graph_path}")
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    from nuthatch.clustering.protocol import ClusteringRequest, Rigor
    from nuthatch.clustering.router import ClusteringRouter
    from nuthatch.graph.io import load_graph, save_graph

    layout = _resolve_layout_or_die(args)
    graph_path = layout.kg / "graph" / "graph.json"
    if not graph_path.is_file():
        print(
            f"error: no graph at {graph_path}. Run `nuthatch ingest` and "
            "`nuthatch embed` first, then this command.",
            file=sys.stderr,
        )
        return 2

    import io as _io
    import json as _json

    g = load_graph(graph_path)
    # ClusteringRequest takes a serialised graph snapshot; backends
    # that need the graph deserialise it themselves. We pass the
    # JSON node-link form so the protocol stays backend-agnostic.
    buf = _io.BytesIO()
    import networkx as nx

    payload = nx.node_link_data(g, edges="edges")
    buf.write(_json.dumps(payload).encode("utf-8"))
    request = ClusteringRequest(buf.getvalue(), rigor=Rigor.PRINCIPLED)

    router = ClusteringRouter()
    response = router.cluster(request)

    # Write community membership back onto graph node attributes
    # so downstream render + decay layers can read it without a
    # second clustering pass.
    for node_id, community_id in response.partition.items():
        if node_id in g:
            g.nodes[node_id]["community_id"] = int(community_id)
    save_graph(g, graph_path)

    n_communities = len(set(response.partition.values()))
    print(
        f"[OK] cluster: {len(response.partition)} nodes -> "
        f"{n_communities} communities "
        f"(rigor={response.rigor_used.value}, "
        f"backend={response.backend_used}, "
        f"{response.runtime_seconds:.1f}s)"
    )
    if response.notes:
        print(f"     notes: {response.notes}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    import json as _json
    from collections import defaultdict

    from nuthatch.graph.io import load_graph
    from nuthatch.render.obsidian import export_vault

    layout = _resolve_layout_or_die(args)
    graph_path = layout.kg / "graph" / "graph.json"
    if not graph_path.is_file():
        print(
            f"error: no graph at {graph_path}. Run `nuthatch ingest`, "
            "`nuthatch embed`, and `nuthatch cluster` first.",
            file=sys.stderr,
        )
        return 2

    g = load_graph(graph_path)

    # Build paper_metadata from the .kg/extracted/*.meta.json sidecars
    # written by IngestOrchestrator.
    paper_metadata: dict[str, dict] = {}
    for meta_path in layout.extracted_dir.glob("*.meta.json"):
        try:
            data = _json.loads(meta_path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError:
            continue
        doc_id = str(data.get("doc_id") or meta_path.stem.replace(".meta", ""))
        meta = data.get("metadata") or {}
        if isinstance(meta, dict):
            paper_metadata[doc_id] = meta

    # Build community_membership from the graph nodes' community_id attr.
    community_membership: dict[str, list[str]] = defaultdict(list)
    for node_id, attrs in g.nodes(data=True):
        if not isinstance(node_id, str) or not node_id.startswith("doc::"):
            continue
        cid = attrs.get("community_id")
        if cid is None:
            continue
        doc_id = node_id[len("doc::") :]
        community_membership[str(cid)].append(doc_id)

    result = export_vault(
        layout,
        paper_metadata=paper_metadata,
        community_membership=dict(community_membership) or None,
        source_note="render",
    )
    print(
        f"[OK] render: {result.n_cards} cards, "
        f"{result.n_communities} community pages"
    )
    print(f"     cards: {result.cards_dir}")
    print(f"     communities: {result.communities_dir}")
    print(f"     dashboard: {result.dashboard_path}")
    return 0


def _print_results(results: list[IngestResult]) -> None:
    if not results:
        print("(no files to process)")
        return
    counts: Counter[str] = Counter()
    for r in results:
        counts[r.status.value] += 1
    for status, n in counts.most_common():
        print(f"  {status:11} {n}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
