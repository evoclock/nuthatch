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
from nuthatch.ingest import IngestOrchestrator, IngestResult
from nuthatch.ingest.manifest import IngestStatus, ManifestStore
from nuthatch.ingest.watch import InboxWatcher


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
