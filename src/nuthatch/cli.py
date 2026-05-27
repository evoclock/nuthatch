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
from typing import Any

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
    ingest_p.add_argument(
        "--skip-chandra",
        action="store_true",
        help=(
            "bypass Chandra-OCR routing entirely; route every PDF through "
            "Docling (with EasyOCR fallback for genuinely-scanned PDFs). "
            "Math-heavy docs whose Docling output has broken LaTeX spans "
            "are flagged in <corpus>/.kg/math_retry.jsonl for a later "
            "batch-Chandra patch pass. Use when you want a fast first ingest "
            "and will patch math separately."
        ),
    )
    ingest_p.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "cuda", "mps", "xpu"),
        default=None,
        help=(
            "device for Docling layout / table-structure / OCR models. "
            "Default `auto` lets Docling pick (CUDA on Nvidia, MPS on Apple "
            "Silicon, XPU on Intel, CPU as fallback). Override with `cuda` / "
            "`mps` / `cpu` to force. Equivalent to setting "
            "NUTHATCH_ACCELERATOR in the environment; this flag wins when both "
            "are set."
        ),
    )

    watch_p = subparsers.add_parser("watch", help="long-running: ingest on FS events")
    _add_corpus_arg(watch_p)

    triage_p = subparsers.add_parser(
        "triage",
        help="pre-flight: classify PDFs as PASS / FLAG / DEFER via pdftotext (no GPU)",
    )
    _add_corpus_arg(triage_p)
    triage_p.add_argument(
        "--defer",
        action="store_true",
        help=(
            "auto-move DEFER-classified PDFs to <corpus>/defer/<original_subdir>/ "
            "so the next `nuthatch ingest` run does not pick them up. Without "
            "this flag, triage is read-only and just prints the report."
        ),
    )

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
    embed_p.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "cuda", "mps", "xpu"),
        default=None,
        help=(
            "device for the BGE-M3 sentence-transformer. Default `auto` picks "
            "CUDA on Nvidia, MPS on Apple Silicon, CPU as fallback. Override "
            "with `cuda` / `mps` / `cpu` to force. Equivalent to setting "
            "NUTHATCH_ACCELERATOR in the environment; the CLI flag wins."
        ),
    )

    semex_p = subparsers.add_parser(
        "semantic-extract",
        help=(
            "extract semantic context per document via LLM. Writes one "
            "<doc_id>.concepts.json sidecar carrying summary + topics + "
            "methods + named entities + domain. Idempotent: skips docs "
            "that already have a sidecar (use --force to re-extract)."
        ),
    )
    _add_corpus_arg(semex_p)
    semex_p.add_argument(
        "--backend", default="ollama",
        help="LLM backend (default: ollama)",
    )
    semex_p.add_argument(
        "--model", default="gemini-3-flash-preview:cloud",
        help="model id; default is Gemini Flash cloud for speed",
    )
    semex_p.add_argument(
        "--num-predict", type=int, default=3072,
        help="token budget for the LLM (default: 3072 for the 5-field response)",
    )
    semex_p.add_argument(
        "--max-docs", type=int, default=0,
        help="process only the first N docs (0 = all)",
    )
    semex_p.add_argument(
        "--excerpt-chars", type=int, default=6000,
        help="how much of each body to send to the LLM",
    )
    semex_p.add_argument(
        "--force", action="store_true",
        help="re-extract even if a sidecar already exists",
    )

    graph_p = subparsers.add_parser(
        "graph",
        help=(
            "build the corpus graph from <corpus>/.kg/extracted/. "
            "Entity extraction + node/edge assembly. Folds in semantic "
            "sidecars (when present) so the graph carries topics, methods, "
            "named entities, and summary-similarity edges - intrinsic to "
            "the build, no separate augmentation step."
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
    cluster_p.add_argument(
        "--backend", choices=("sbm", "leiden", "embeddings"), default=None,
        help="force a specific clustering backend, bypassing the rigor "
             "router. Required when running multiple backends in sequence "
             "for comparison. SBM needs graph-tool (conda env); leiden + "
             "embeddings work in the venv.",
    )
    cluster_p.add_argument(
        "--output-suffix", default=None,
        help="also copy the persisted communities.json to "
             "communities_<suffix>.json after the run, so multiple backends "
             "can co-exist on disk. Same for community_centroids.npy. The "
             "canonical communities.json is preserved.",
    )
    cluster_p.add_argument(
        "--relabel-llm", action="store_true",
        help="after clustering, replace the heuristic word-frequency "
             "labels with topical 2-4 word names produced by an LLM "
             "(default backend: ollama, default model: granite3-dense:8b). "
             "Writes to canonical communities.json and any "
             "communities_<suffix>.json siblings produced this run.",
    )
    cluster_p.add_argument(
        "--relabel-backend", default="ollama",
        help="LLM backend used by --relabel-llm. Supported: ollama, "
             "openai, anthropic. Default: ollama (local, no API key).",
    )
    cluster_p.add_argument(
        "--relabel-model", default=None,
        help="LLM model id used by --relabel-llm. Default depends on "
             "backend; for ollama: granite3-dense:8b.",
    )

    eval_p = subparsers.add_parser(
        "eval",
        help=(
            "quality evaluations across pipeline phases. Subcommands: "
            "`graph`, `cluster`, `ragas`. Each writes a markdown report "
            "to `pipeline_output/`. Slow when LLM judges are enabled; "
            "use --skip-judge / --skip-ragas for fast smoke checks."
        ),
    )
    eval_sub = eval_p.add_subparsers(dest="eval_command", required=True)

    eval_graph_p = eval_sub.add_parser(
        "graph",
        help=(
            "structural sanity + LLM-judged extraction fidelity for the "
            "corpus graph. Reports node/edge counts by type, density, "
            "component fragmentation, and (optionally) per-doc precision."
        ),
    )
    _add_corpus_arg(eval_graph_p)
    eval_graph_p.add_argument("--sample-docs", type=int, default=100)
    eval_graph_p.add_argument("--seed", type=int, default=0)
    eval_graph_p.add_argument("--judge-backend", default="ollama")
    eval_graph_p.add_argument("--judge-model", default="granite3-dense:8b")
    eval_graph_p.add_argument("--judge-num-predict", type=int, default=1024)
    eval_graph_p.add_argument("--skip-judge", action="store_true")
    eval_graph_p.add_argument("--out-dir", default="pipeline_output")

    eval_cluster_p = eval_sub.add_parser(
        "cluster",
        help=(
            "structural sanity (modularity, size distribution, etc.) + "
            "LLM-judged community coherence for a given communities file. "
            "Use --communities <suffix> to pick a specific backend's "
            "output (communities_sbm.json, communities_leiden.json, ...)."
        ),
    )
    _add_corpus_arg(eval_cluster_p)
    eval_cluster_p.add_argument(
        "--communities", default=None,
        help="suffix of the communities file to evaluate "
             "(omit for default communities.json)",
    )
    eval_cluster_p.add_argument("--sample-communities", type=int, default=100)
    eval_cluster_p.add_argument("--docs-per-community", type=int, default=5)
    eval_cluster_p.add_argument("--seed", type=int, default=0)
    eval_cluster_p.add_argument("--judge-backend", default="ollama")
    eval_cluster_p.add_argument("--judge-model", default="granite3-dense:8b")
    eval_cluster_p.add_argument("--judge-num-predict", type=int, default=1024)
    eval_cluster_p.add_argument("--skip-judge", action="store_true")
    eval_cluster_p.add_argument("--out-dir", default="pipeline_output")

    eval_ragas_p = eval_sub.add_parser(
        "ragas",
        help=(
            "RAGAS-based retrieval + answer eval. Builds (or reuses) a "
            "synthetic test set, runs the retriever + LLM answerer per "
            "question, scores with RAGAS metrics, writes report."
        ),
    )
    _add_corpus_arg(eval_ragas_p)
    eval_ragas_p.add_argument("--n-questions", type=int, default=100)
    eval_ragas_p.add_argument("--per-doc-cap", type=int, default=3)
    eval_ragas_p.add_argument("--k", type=int, default=5)
    eval_ragas_p.add_argument("--seed", type=int, default=0)
    eval_ragas_p.add_argument("--gen-backend", default="ollama")
    eval_ragas_p.add_argument(
        "--gen-model", default="gemini-3-flash-preview:cloud",
        help="generator + answerer model. Default is Gemini Flash cloud "
             "(fast + JSON-compliant for testset generation, paired with "
             "local granite3-dense:8b as the cross-family judge to reduce "
             "self-preference bias).",
    )
    eval_ragas_p.add_argument("--gen-num-predict", type=int, default=2048)
    eval_ragas_p.add_argument("--judge-backend", default="ollama")
    eval_ragas_p.add_argument(
        "--judge-model", default="granite3-dense:8b",
        help="structured-output local judge (IBM Granite 3 dense 8B). "
             "Non-reasoning so it does not burn the token budget on a "
             "thinking trace before emitting JSON content. Different "
             "family from Gemini generator to reduce self-preference bias.",
    )
    eval_ragas_p.add_argument("--judge-num-predict", type=int, default=1024)
    eval_ragas_p.add_argument("--out-dir", default="pipeline_output")
    eval_ragas_p.add_argument("--skip-ragas", action="store_true")
    eval_ragas_p.add_argument("--measure-rerank", action="store_true")
    eval_ragas_p.add_argument("--reuse-testset", nargs="+", default=None)

    viz_p = subparsers.add_parser(
        "viz",
        help=(
            "graph + community visualisations. Subcommand: `d3` "
            "(server-laid-out D3 canvas with optional community overlays)."
        ),
    )
    viz_sub = viz_p.add_subparsers(dest="viz_command", required=True)

    viz_d3_p = viz_sub.add_parser(
        "d3",
        help=(
            "render an interactive D3 + canvas HTML of the corpus graph. "
            "Pass --communities <suffix> to colour by one backend, or "
            "--communities all to emit one HTML per available "
            "communities_*.json side-by-side."
        ),
    )
    _add_corpus_arg(viz_d3_p)
    viz_d3_p.add_argument(
        "--communities", default=None,
        help="suffix of communities file to overlay "
             "('sbm', 'leiden', 'embeddings', ...), or 'all' to render "
             "one HTML per available backend output. Omit for an "
             "uncoloured baseline HTML.",
    )
    viz_d3_p.add_argument("--filter-types", default=None,
        help="comma-separated entity types to keep "
             "(e.g. document,topic)")
    viz_d3_p.add_argument("--max-nodes", type=int, default=3000)
    viz_d3_p.add_argument("--layout", default="forceatlas2",
        choices=("forceatlas2", "spring", "kamada_kawai"))
    viz_d3_p.add_argument("--layout-iterations", type=int, default=200)
    viz_d3_p.add_argument("--out-dir", default="pipeline_output")

    render_p = subparsers.add_parser(
        "render",
        help=(
            "regenerate Obsidian-compatible cards/, communities/, "
            "dashboard.md, index.md from current corpus state."
        ),
    )
    _add_corpus_arg(render_p)

    serve_p = subparsers.add_parser(
        "serve",
        help=(
            "start an MCP stdio server for the corpus. Exposes "
            "corpus_search, community_search/brief/get/core_nodes/"
            "hierarchy, card_get, subgraph_extract, token_econ_report. "
            "Speak JSON-RPC on stdin/stdout — typically spawned by an "
            "agent host (Claude Code, Codex, ...) via mcpServers config."
        ),
    )
    _add_corpus_arg(serve_p)

    publish_p = subparsers.add_parser(
        "publish",
        help=(
            "promote a corpus's agent-facing surface (cards, "
            "communities, graph, chroma, .obsidian config, eval "
            "reports) into a shareable KB directory."
        ),
    )
    _add_corpus_arg(publish_p)
    publish_p.add_argument(
        "--to", type=str, required=True,
        help="destination directory for the published KB.",
    )
    publish_p.add_argument(
        "--name", type=str, default=None,
        help=(
            "KB name used in generated README / AGENTS.md / config. "
            "Defaults to the destination directory's basename."
        ),
    )
    publish_p.add_argument(
        "--license", type=str, default="CC-BY-4.0",
        help=(
            "SPDX license identifier for the published KB. "
            "Defaults to CC-BY-4.0. Use PROPRIETARY for private "
            "distribution."
        ),
    )
    publish_p.add_argument(
        "--no-chroma", action="store_true",
        help=(
            "skip the chroma vector store (large; contains chunk text "
            "which may be license-mixed for non-CC-BY corpora)."
        ),
    )
    publish_p.add_argument(
        "--include-bodies", action="store_true",
        help=(
            "include .kg/extracted/ full-text bodies. Only safe for "
            "corpora where every source is permissively-licensed."
        ),
    )
    publish_p.add_argument(
        "--no-eval", action="store_true",
        help="skip copying eval reports from docs/.",
    )
    publish_p.add_argument(
        "--no-d3-html", action="store_true",
        help="skip the D3 topology HTML.",
    )

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
    if args.subcommand == "triage":
        return _cmd_triage(args)
    if args.subcommand == "status":
        return _cmd_status(args)
    if args.subcommand == "corpus" and args.corpus_command == "list":
        return _cmd_corpus_list()
    if args.subcommand == "token-report":
        return _cmd_token_report(args)
    if args.subcommand == "decay":
        return _cmd_decay(args)
    if args.subcommand == "semantic-extract":
        return _cmd_semantic_extract(args)
    if args.subcommand == "embed":
        return _cmd_embed(args)
    if args.subcommand == "graph":
        return _cmd_graph(args)
    if args.subcommand == "cluster":
        return _cmd_cluster(args)
    if args.subcommand == "eval":
        if args.eval_command == "graph":
            return _cmd_eval_graph(args)
        if args.eval_command == "cluster":
            return _cmd_eval_cluster(args)
        if args.eval_command == "ragas":
            return _cmd_eval_ragas(args)
    if args.subcommand == "viz" and args.viz_command == "d3":
        return _cmd_viz_d3(args)
    if args.subcommand == "render":
        return _cmd_render(args)
    if args.subcommand == "serve":
        return _cmd_serve(args)
    if args.subcommand == "publish":
        return _cmd_publish(args)

    parser.print_help()
    return 2


def _cmd_init(args: argparse.Namespace) -> int:
    layout = init_corpus(args.path)
    print(f"[OK] initialised corpus at {layout.root}")
    print(f"     marker:   {layout.kg}")
    print(f"     inbox:    {layout.inbox}")
    print(f"     processed: {layout.processed}")
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
    # Accelerator must be set BEFORE any extractor / Docling import so
    # the layout / OCR models load onto the requested device. The CLI
    # flag wins over an inherited NUTHATCH_ACCELERATOR env value so
    # operators can override per-run without unsetting the env.
    import os
    accel = getattr(args, "accelerator", None)
    if accel:
        os.environ["NUTHATCH_ACCELERATOR"] = accel
    effective_accel = os.environ.get("NUTHATCH_ACCELERATOR", "auto")
    print(f"[ingest] accelerator: {effective_accel}", flush=True)

    layout = _resolve_layout_or_die(args)
    skip_chandra = bool(getattr(args, "skip_chandra", False))
    orchestrator = IngestOrchestrator(layout, skip_chandra=skip_chandra)
    if skip_chandra:
        print(
            "[ingest] --skip-chandra: Chandra disabled. Math-heavy docs "
            "will be flagged in <corpus>/.kg/math_retry.jsonl for later "
            "batch-Chandra patching.",
            flush=True,
        )

    # Show the operator something IMMEDIATELY, before paying the
    # extractor import cost (Docling + torch can take 30s-2min to
    # load on first run). Without this the log is silent for ages
    # and looks hung.
    print(f"[ingest] corpus root: {layout.root}", flush=True)
    sources = orchestrator._gather_source_files()
    print(
        f"[ingest] scanned {len(sources)} file(s) under the corpus root "
        "(reserved dirs skipped)",
        flush=True,
    )
    if not sources:
        print("[ingest] nothing to do.", flush=True)
        return 0
    print(
        "[ingest] loading extractor (Docling first-run model load can "
        "take 30s-2min; subsequent files are fast)...",
        flush=True,
    )

    def _on_progress(i: int, total: int, result: IngestResult) -> None:
        # Per-file line so a 100+-file run shows live progress, not
        # a silent wait followed by a single summary.
        reason_suffix = f" ({result.reason})" if result.reason else ""
        print(
            f"[{i:>4}/{total}] {result.status.value:<11} "
            f"{result.source_filename}{reason_suffix}",
            flush=True,
        )

    results = orchestrator.ingest_corpus(on_progress=_on_progress)
    print("---", flush=True)
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


def _cmd_triage(args: argparse.Namespace) -> int:
    """Pre-flight: classify candidate PDFs via pdftotext (no GPU).

    Output:
      - Per-class list of PDFs with reason
      - Aggregate summary
      - When `--defer` is set: auto-moves DEFER PDFs to
        `<corpus>/defer/<original_subdir>/` so the next
        `nuthatch ingest` skips them.
    """
    from nuthatch.ingest.triage import (
        TriageClass,
        auto_defer,
        triage_corpus,
    )

    layout = _resolve_layout_or_die(args)
    results = triage_corpus(layout)
    if not results:
        print("[triage] no PDFs in scan path (every input is already processed?)")
        return 0

    by_class: dict[str, list[Any]] = {c.value: [] for c in TriageClass}
    for r in results:
        by_class[r.classification.value].append(r)

    print(f"[triage] {len(results)} PDFs across the scan path")
    print()
    for cls in (TriageClass.PASS, TriageClass.FLAG, TriageClass.DEFER, TriageClass.UNKNOWN):
        items = by_class[cls.value]
        if not items:
            continue
        print(f"{cls.value} ({len(items)}):")
        for r in items:
            print(f"  {r.path.relative_to(layout.root)}")
            if cls is not TriageClass.PASS:
                print(f"      reason: {r.reason}")
        print()

    if args.defer:
        moved = auto_defer(results, layout)
        print(f"[--defer] moved {len(moved)} DEFER PDFs to {layout.root / 'defer'}/")
    elif by_class[TriageClass.DEFER.value]:
        print(
            "[hint] re-run with `--defer` to auto-move the "
            f"{len(by_class[TriageClass.DEFER.value])} DEFER PDFs out of "
            "the scan path so `nuthatch ingest` skips them."
        )
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


def _cmd_semantic_extract(args: argparse.Namespace) -> int:
    """Run LLM-based semantic-context extraction per doc.

    Writes one `<corpus>/.kg/extracted/<doc_id>.concepts.json` sidecar
    carrying summary + topics + methods + named_entities + domain.
    Idempotent: skips docs whose sidecar already exists unless --force.

    Delegates the heavy lifting to `semantic_extract.extract.main` so
    the CLI stays a thin wrapper. The argparse signature here mirrors
    that module so the user sees the same flags either way.
    """
    from nuthatch.semantic_extract.extract import main as _extract_main

    forwarded = ["--corpus", args.corpus] if getattr(args, "corpus", None) else []
    forwarded += [
        "--backend", args.backend,
        "--model", args.model,
        "--num-predict", str(args.num_predict),
        "--max-docs", str(args.max_docs),
        "--excerpt-chars", str(args.excerpt_chars),
    ]
    if getattr(args, "force", False):
        forwarded.append("--force")
    return _extract_main(forwarded)


def _cmd_embed(args: argparse.Namespace) -> int:
    # Set NUTHATCH_ACCELERATOR BEFORE any heavy import so the Embedder
    # constructor (which reads the env var) sees it. The CLI flag wins
    # over an inherited env value.
    import os
    accel = getattr(args, "accelerator", None)
    if accel:
        os.environ["NUTHATCH_ACCELERATOR"] = accel
    effective_accel = os.environ.get("NUTHATCH_ACCELERATOR", "auto")
    print(f"[embed] accelerator: {effective_accel}", flush=True)

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
    if result.n_concept_sidecars > 0:
        print(
            f"     semantic: {result.n_concept_sidecars} sidecars folded in "
            f"({result.n_concept_nodes} concept nodes, "
            f"{result.n_mention_edges} mention edges, "
            f"{result.n_semantic_edges} shares_summary_with edges)"
        )
    else:
        print(
            "     semantic: no <doc_id>.concepts.json sidecars present; "
            "run `nuthatch semantic-extract` to enable conceptual + "
            "semantic-edge augmentation."
        )
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

    import json as _json

    g = load_graph(graph_path)
    import networkx as nx

    # The three backends consume different snapshot shapes:
    #   - SBM, Leiden: a node-link graph JSON. We feed them the
    #     DOC-DOC BIPARTITE PROJECTION of the augmented graph, not
    #     the raw augmented graph itself. The augmented graph is ~97%
    #     entity nodes (authors, citations, topics, methods) and 86%
    #     of edges are `co_mentioned_in` between entities; running
    #     SBM/Leiden on it yields communities of co-occurring
    #     entities, not of papers sharing concepts. The projection
    #     keeps the entity-mediated signal as edge weight while
    #     making documents the unit of clustering.
    #   - Embeddings (k-means): {node_ids, embeddings} JSON, no graph.
    # Keep the CLI as the only place that knows about both shapes so
    # the backends stay agnostic to corpus storage. The router default
    # (rigor=PRINCIPLED) only ever picks SBM or Leiden, never
    # embeddings, so it always wants the graph snapshot.
    backend_override = getattr(args, "backend", None)
    if backend_override == "embeddings":
        # Pull doc-level centroids from Chroma so the embeddings
        # backend has something to k-means over. Mean-pool each
        # document's chunk vectors into one doc vector.
        node_ids, embeddings = _compute_doc_centroids(layout=layout, graph=g)
        if not node_ids:
            print(
                "error: --backend embeddings requires the embed stage "
                "to have run; no chunks found in Chroma.",
                file=sys.stderr,
            )
            return 2
        snapshot_payload = {"node_ids": node_ids, "embeddings": embeddings}
    else:
        # Project to doc-doc graph before clustering. The projection
        # is intrinsic to the cluster CLI's input prep (not a separate
        # phase) so re-runs are idempotent and the user can't forget.
        from nuthatch.clustering.projection import project_to_doc_doc
        g_for_clustering = project_to_doc_doc(g)
        print(
            f"[cluster] projected to doc-doc: "
            f"{g_for_clustering.number_of_nodes()} doc nodes, "
            f"{g_for_clustering.number_of_edges()} weighted edges "
            f"(from augmented graph: {g.number_of_nodes()} nodes, "
            f"{g.number_of_edges()} edges)"
        )
        snapshot_payload = nx.node_link_data(g_for_clustering, edges="edges")

    snapshot_bytes = _json.dumps(snapshot_payload).encode("utf-8")
    request = ClusteringRequest(snapshot_bytes, rigor=Rigor.PRINCIPLED)

    # --backend forces a specific backend bypassing the router. Used
    # for side-by-side comparisons across SBM / Leiden / embeddings.
    # Default path uses the rigor router (highest-rigor available).
    if backend_override:
        if backend_override == "sbm":
            from nuthatch.clustering.backends.sbm import SBMBackend
            response = SBMBackend().cluster(request)
        elif backend_override == "leiden":
            from nuthatch.clustering.backends.leiden import LeidenBackend
            response = LeidenBackend().cluster(request)
        elif backend_override == "embeddings":
            from nuthatch.clustering.backends.embeddings import EmbeddingsBackend
            response = EmbeddingsBackend().cluster(request)
        else:
            print(
                f"error: unknown --backend {backend_override!r}",
                file=sys.stderr,
            )
            return 2
        print(f"[cluster] backend forced: {backend_override} "
              f"(rigor={response.rigor_used.value})")
    else:
        router = ClusteringRouter()
        response = router.cluster(request)

    # Write community membership back onto graph node attributes
    # so downstream render + decay layers can read it without a
    # second clustering pass.
    for node_id, community_id in response.partition.items():
        if node_id in g:
            g.nodes[node_id]["community_id"] = int(community_id)
    save_graph(g, graph_path)

    # Persist the canonical community index at <corpus>/.kg/communities.json.
    # This is what the MCP server's community-aware tools read at query
    # time: per-doc community_id, the nested hierarchy chain (SBM), per-
    # community core nodes, and human-readable labels. Centroids (for
    # semantic community.search) are computed below when embeddings are
    # available.
    from nuthatch.clustering.persist import persist_communities

    paper_metadata = _load_paper_metadata(layout)
    centroids = _compute_community_centroids(
        layout=layout,
        partition=response.partition,
    )
    index_path = persist_communities(
        layout=layout,
        response=response,
        graph=g,
        paper_metadata=paper_metadata,
        centroids=centroids,
    )

    n_communities = len(set(response.partition.values()))
    print(
        f"[OK] cluster: {len(response.partition)} nodes -> "
        f"{n_communities} communities "
        f"(rigor={response.rigor_used.value}, "
        f"backend={response.backend_used}, "
        f"{response.runtime_seconds:.1f}s)"
    )
    print(f"     index:  {index_path.relative_to(layout.root)}")
    if centroids:
        print(f"     centroids: {len(centroids)} communities x "
              f"{len(next(iter(centroids.values())))} dims")
    if response.notes:
        print(f"     notes:  {response.notes}")

    # --output-suffix copies the persisted index + centroids under a
    # suffixed name so multiple backends can co-exist on disk for
    # side-by-side comparison. We COPY (not rename) so the canonical
    # `communities.json` / `community_centroids.npy` always remains
    # in place; the MCP server's community_* tools and publish.py
    # both depend on the canonical name being present.
    suffix = getattr(args, "output_suffix", None)
    if suffix:
        import shutil

        suffixed_index = layout.kg / f"communities_{suffix}.json"
        shutil.copy2(index_path, suffixed_index)
        print(f"     copied:  {suffixed_index.relative_to(layout.root)}")
        centroids_src = layout.kg / "community_centroids.npy"
        if centroids_src.is_file():
            suffixed_centroids = (
                layout.kg / f"community_centroids_{suffix}.npy"
            )
            shutil.copy2(centroids_src, suffixed_centroids)
            print(f"     copied:  "
                  f"{suffixed_centroids.relative_to(layout.root)}")

    # --relabel-llm rewrites the heuristic word-frequency labels with
    # topical 2-4 word names produced by an LLM. Applied to canonical
    # AND every suffixed sibling produced this run so all on-disk
    # views stay consistent. Failure for a single community keeps the
    # old label for that one; failure to build the LLM (model not
    # installed, ollama down) is reported and skipped.
    if getattr(args, "relabel_llm", False):
        from nuthatch.clustering.relabel import (
            DEFAULT_RELABEL_MODEL,
            build_default_invoke,
            relabel_communities,
        )

        backend_name = getattr(args, "relabel_backend", "ollama")
        model_name = (
            getattr(args, "relabel_model", None) or DEFAULT_RELABEL_MODEL
        )
        targets: list[Path] = [index_path]
        if suffix:
            suffixed_target = layout.kg / f"communities_{suffix}.json"
            if suffixed_target.is_file():
                targets.append(suffixed_target)
        cards_dir = layout.root / "cards"
        try:
            invoke = build_default_invoke(
                backend=backend_name, model=model_name,
            )
        except (ImportError, ValueError) as exc:
            print(
                f"     relabel: skipped (could not build {backend_name}/"
                f"{model_name}: {exc!s})"
            )
        else:
            print(
                f"     relabel: {backend_name}/{model_name} over "
                f"{len(targets)} file(s)"
            )
            diffs = relabel_communities(
                targets, cards_dir, llm_invoke=invoke,
            )
            for path, per_file in diffs.items():
                rel = path.relative_to(layout.root)
                print(f"     relabel: {rel}")
                for cid, (old, new) in sorted(per_file.items()):
                    if old != new:
                        print(f"        [{cid:>3}] {old!r}")
                        print(f"              -> {new!r}")
    return 0


def _load_paper_metadata(layout: Any) -> dict[str, dict[str, Any]]:
    """Read every .kg/extracted/<doc_id>.meta.json into a dict keyed by doc_id.

    Used by persist_communities to generate per-community labels from
    member titles. Returns an empty dict when no extracted metadata
    is present (cluster-without-ingest, tests).
    """
    extracted = layout.extracted_dir
    if not extracted.is_dir():
        return {}
    import json as _json
    out: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(extracted.glob("*.meta.json")):
        try:
            data = _json.loads(meta_path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError:
            continue
        doc_id = data.get("doc_id") or meta_path.stem.removesuffix(".meta")
        meta_block = data.get("metadata") or {}
        if doc_id:
            out[str(doc_id)] = meta_block
    return out


def _compute_doc_centroids(
    *,
    layout: Any,
    graph: Any,
) -> tuple[list[str], list[list[float]]]:
    """Mean-pool chunk embeddings per document into one vector per doc.

    Used by `--backend embeddings` to build the cluster request's
    `{node_ids, embeddings}` snapshot. Only documents that (a) appear
    as document nodes in the graph AND (b) have at least one chunk in
    the Chroma store contribute a centroid; everything else is dropped
    silently so partial-state corpora don't error.

    Returns parallel lists: `node_ids` carries the graph's full
    `doc::<bare_id>` keys so the partition the backend returns lines
    up with the graph's node namespace.
    """
    try:
        from nuthatch.embed.store import ChromaVectorStore
    except ImportError:
        return [], []
    try:
        store = ChromaVectorStore(layout.embeddings_dir, collection_name="chunks")
        coll = store._ensure_collection()
    except Exception:
        return [], []

    # Collect graph document nodes; index by bare id so a Chroma
    # query keyed on doc_id maps back to the graph's `doc::<id>` key.
    doc_nodes_by_bare: dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "document":
            bare = str(node).split("::", 1)[-1]
            doc_nodes_by_bare[bare] = str(node)

    node_ids: list[str] = []
    embeddings: list[list[float]] = []
    for bare_id, graph_node_id in doc_nodes_by_bare.items():
        try:
            got = coll.get(where={"doc_id": str(bare_id)}, include=["embeddings"])
        except Exception:
            continue
        raw_vecs = got.get("embeddings")
        if raw_vecs is None:
            continue
        vecs = list(raw_vecs)
        if not vecs:
            continue
        n = len(vecs)
        dim = len(vecs[0])
        centroid = [sum(v[d] for v in vecs) / n for d in range(dim)]
        node_ids.append(graph_node_id)
        embeddings.append(centroid)
    return node_ids, embeddings


def _compute_community_centroids(
    *,
    layout: Any,
    partition: dict[str, int],
) -> dict[int, list[float]]:
    """Average per-doc chunk embeddings into per-community centroids.

    Powers `community_search` (semantic match query -> community).
    Pulls embeddings directly from the Chroma collection via a
    `where={"doc_id": ...}` filter so we don't need a dedicated
    store API. Returns an empty dict when the embed stage hasn't
    run or when Chroma is not reachable; the persistence layer
    handles the absence gracefully (community_search will be
    unavailable, but every other tool still works).
    """
    try:
        from nuthatch.embed.store import ChromaVectorStore
    except ImportError:
        return {}
    try:
        # Collection name must match what `embed_corpus` writes to;
        # see `nuthatch.embed.store` default (`"chunks"`). Using the
        # wrong name silently returns 0 vectors and the centroids
        # file ends up empty.
        store = ChromaVectorStore(layout.embeddings_dir, collection_name="chunks")
        coll = store._ensure_collection()
    except Exception:
        return {}

    from collections import defaultdict

    by_community: dict[int, list[list[float]]] = defaultdict(list)
    for doc_id, community_id in partition.items():
        try:
            got = coll.get(where={"doc_id": str(doc_id)}, include=["embeddings"])
        except Exception:
            continue
        raw_vecs = got.get("embeddings")
        # Chroma returns a numpy array; the truthiness check needs care.
        if raw_vecs is None:
            continue
        vecs = list(raw_vecs)
        if not vecs:
            continue
        n = len(vecs)
        dim = len(vecs[0])
        # Average doc's chunk embeddings into one doc-level vector,
        # then aggregate doc-vectors per community for the final mean.
        doc_centroid = [sum(v[d] for v in vecs) / n for d in range(dim)]
        by_community[int(community_id)].append(doc_centroid)

    out: dict[int, list[float]] = {}
    for cid, doc_vectors in by_community.items():
        if not doc_vectors:
            continue
        dim = len(doc_vectors[0])
        n = len(doc_vectors)
        out[cid] = [sum(v[d] for v in doc_vectors) / n for d in range(dim)]
    return out


def _cmd_eval_graph(args: argparse.Namespace) -> int:
    """Forward to `nuthatch.eval.graph_quality.main` with built argv."""
    from nuthatch.eval.graph_quality import main as _eval_main

    forwarded = ["--corpus", args.corpus] if getattr(args, "corpus", None) else []
    forwarded += [
        "--sample-docs", str(args.sample_docs),
        "--seed", str(args.seed),
        "--judge-backend", args.judge_backend,
        "--judge-model", args.judge_model,
        "--judge-num-predict", str(args.judge_num_predict),
        "--out-dir", args.out_dir,
    ]
    if args.skip_judge:
        forwarded.append("--skip-judge")
    return _eval_main(forwarded)


def _cmd_eval_cluster(args: argparse.Namespace) -> int:
    """Forward to `nuthatch.eval.cluster_quality.main` with built argv.

    `--communities <suffix>` selects which backend's output to score;
    omit to score the default `communities.json`.
    """
    from nuthatch.eval.cluster_quality import main as _eval_main

    forwarded = ["--corpus", args.corpus] if getattr(args, "corpus", None) else []
    if args.communities:
        forwarded += ["--communities", args.communities]
    forwarded += [
        "--sample-communities", str(args.sample_communities),
        "--docs-per-community", str(args.docs_per_community),
        "--seed", str(args.seed),
        "--judge-backend", args.judge_backend,
        "--judge-model", args.judge_model,
        "--judge-num-predict", str(args.judge_num_predict),
        "--out-dir", args.out_dir,
    ]
    if args.skip_judge:
        forwarded.append("--skip-judge")
    return _eval_main(forwarded)


def _cmd_eval_ragas(args: argparse.Namespace) -> int:
    """Forward to `nuthatch.eval.ragas_runner.main` with built argv."""
    from nuthatch.eval.ragas_runner import main as _eval_main

    forwarded = ["--corpus", args.corpus] if getattr(args, "corpus", None) else []
    forwarded += [
        "--n-questions", str(args.n_questions),
        "--per-doc-cap", str(args.per_doc_cap),
        "--k", str(args.k),
        "--seed", str(args.seed),
        "--gen-backend", args.gen_backend,
        "--gen-model", args.gen_model,
        "--gen-num-predict", str(args.gen_num_predict),
        "--judge-backend", args.judge_backend,
        "--judge-model", args.judge_model,
        "--judge-num-predict", str(args.judge_num_predict),
        "--out-dir", args.out_dir,
    ]
    if args.skip_ragas:
        forwarded.append("--skip-ragas")
    if args.measure_rerank:
        forwarded.append("--measure-rerank")
    if args.reuse_testset:
        forwarded += ["--reuse-testset", *args.reuse_testset]
    return _eval_main(forwarded)


def _cmd_viz_d3(args: argparse.Namespace) -> int:
    """Forward to `nuthatch.viz.d3_renderer.main` with built argv.

    Thin wrapper so the CLI stays the user-facing surface; the
    rendering logic lives in the module.
    """
    from nuthatch.viz.d3_renderer import main as _viz_main

    forwarded = ["--corpus", args.corpus] if getattr(args, "corpus", None) else []
    if args.communities:
        forwarded += ["--communities", args.communities]
    if args.filter_types:
        forwarded += ["--filter-types", args.filter_types]
    forwarded += [
        "--max-nodes", str(args.max_nodes),
        "--layout", args.layout,
        "--layout-iterations", str(args.layout_iterations),
        "--out-dir", args.out_dir,
    ]
    return _viz_main(forwarded)


def _cmd_render(args: argparse.Namespace) -> int:
    import json as _json

    from nuthatch.clustering.persist import load_community_index
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

    # Build paper_metadata from the .kg/extracted/*.meta.json sidecars.
    paper_metadata: dict[str, dict] = {}
    for meta_path in layout.extracted_dir.glob("*.meta.json"):
        try:
            data = _json.loads(meta_path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError:
            continue
        doc_id = str(data.get("doc_id") or meta_path.stem.replace(".meta", ""))
        meta = data.get("metadata") or {}
        if isinstance(meta, dict):
            paper_metadata[doc_id] = dict(meta)

    # Merge topics, methods, summary from .concepts.json sidecars.
    for concepts_path in layout.extracted_dir.glob("*.concepts.json"):
        try:
            concepts = _json.loads(concepts_path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError:
            continue
        doc_id = str(
            concepts.get("doc_id")
            or concepts_path.stem.replace(".concepts", "")
        )
        if doc_id in paper_metadata:
            for field in ("topics", "methods", "summary"):
                val = concepts.get(field)
                if val:
                    paper_metadata[doc_id][field] = val

    # Load community index from persisted file; prefer SBM backend.
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

    # Build community_membership from the persisted index.
    community_membership: dict[str, list[str]] = {}
    if community_index is not None:
        for cid_int, members in community_index.members.items():
            doc_ids = [
                m[len("doc::"):] if m.startswith("doc::") else m
                for m in members
            ]
            community_membership[str(cid_int)] = doc_ids

    result = export_vault(
        layout,
        paper_metadata=paper_metadata,
        community_membership=community_membership or None,
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


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the MCP stdio server against the resolved corpus.

    Stdin/stdout speak JSON-RPC. Typically launched as a subprocess
    by an agent host (Claude Code via `mcpServers` config, etc.) —
    running this in a terminal interactively works but isn't useful;
    the process just waits for JSON-RPC on stdin.

    All MCP tools (corpus_search, community_*, card_get, ...) are
    served from the corpus indexes: chroma store under
    `.kg/embeddings/`, graph under `.kg/graph/graph.json`, community
    indexes under `.kg/communities_*.json`.
    """
    from nuthatch.embed.embed import Embedder
    from nuthatch.embed.store import ChromaVectorStore
    from nuthatch.graph.io import load_graph
    from nuthatch.mcp.server import NuthatchMCPServer
    from nuthatch.retrieve.vector import VectorRetriever
    from nuthatch.token_econ.counterfactual import build_default_estimator

    layout = _resolve_layout_or_die(args)

    if not layout.embeddings_dir.is_dir():
        print(
            f"error: no chroma store at {layout.embeddings_dir}. "
            "If this is a published KB, unpack the archive first:\n"
            f"  mkdir -p {layout.kg} && tar xzf "
            f"{layout.root / 'embeddings.tar.gz'} -C {layout.kg}",
            file=sys.stderr,
        )
        return 2

    graph_path = layout.kg / "graph" / "graph.json"
    if not graph_path.is_file():
        print(
            f"error: no graph at {graph_path}. Run `nuthatch graph` "
            "to build it (or, for a published KB, the missing file "
            "indicates an incomplete publish).",
            file=sys.stderr,
        )
        return 2

    store = ChromaVectorStore(root=layout.embeddings_dir)
    retriever = VectorRetriever(store, embedder=Embedder())

    def _graph_loader():
        return load_graph(graph_path)

    # Bind a TokenLog so `token_econ_report` works. The log is append-only;
    # missing file is the empty-history case. Path is stable so reports
    # across MCP-server restarts aggregate against the same store.
    token_log_path = layout.kg / "token_log.jsonl"
    token_log = TokenLog(token_log_path)

    estimator = build_default_estimator(
        layout,
        chunks_provider=store.iter_chunks,
    )

    server = NuthatchMCPServer(
        layout,
        retriever=retriever,
        graph_loader=_graph_loader,
        token_log=token_log,
        counterfactual_estimator=estimator,
    )
    server.run()
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    """Promote a corpus's agent-facing surface to a publishable KB
    directory. Heavy lifting lives in `nuthatch.publish.publish_corpus`;
    this wrapper resolves the corpus + optional tool-repo root, then
    pretty-prints the result."""
    from nuthatch.publish import publish_corpus

    layout = _resolve_layout_or_die(args)
    dest = Path(args.to).expanduser().resolve()

    # The tool repo root carries `docs/eval-*.md` reports. We detect it
    # by walking up from this file; fall back to None which disables the
    # eval-copy step gracefully.
    try:
        tool_repo_root = Path(__file__).resolve().parents[2]
        if not (tool_repo_root / "docs").is_dir():
            tool_repo_root = None
    except Exception:
        tool_repo_root = None

    print(f"[publish] corpus: {layout.root}")
    print(f"[publish] dest:   {dest}")
    print(f"[publish] license: {args.license}")

    result = publish_corpus(
        corpus_root=layout.root,
        dest=dest,
        kb_name=args.name,
        license_spdx=args.license,
        include_chroma=not args.no_chroma,
        include_bodies=args.include_bodies,
        include_eval=not args.no_eval,
        include_d3_html=not args.no_d3_html,
        tool_repo_root=tool_repo_root,
    )

    print(f"[OK] publish: {result.n_cards} cards, "
          f"{result.n_community_pages} community pages")
    print(f"     dest: {result.dest}")
    print(f"     bytes written: {result.bytes_written:,}")
    print(f"     chroma included: {result.included_chroma}")
    print(f"     bodies included: {result.included_bodies}")
    print(f"     eval reports included: {result.included_eval}")
    print(f"     manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
