# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Filesystem-event-driven ingest for `nuthatch watch`.

Uses `watchdog` to listen for file-create / file-move events under
the entire corpus root (skipping reserved nuthatch-managed
subdirs). Each event schedules a debounced re-run of
`IngestOrchestrator.ingest_corpus()` (debouncing matters because
batch copy operations fire one event per file; we want to handle
them as a single ingest pass instead of N separate ones).

This is the long-running mode. One-shot ingest stays available via
`IngestOrchestrator.ingest_corpus()` invoked from the CLI's
`nuthatch ingest` subcommand. Watch mode is `nuthatch watch`.

Reserved-dir filter: events whose path starts under
`CORPUS_RESERVED_DIRS` (`.kg/`, `papers/`, `quarantine/`, `cards/`,
`html/`, `communities/`, `graph/`, `exports/`, `reports/`) are
ignored. This prevents the watcher from re-triggering ingest on
files the pipeline itself just wrote.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from nuthatch.corpus.layout import CORPUS_RESERVED_DIRS
from nuthatch.ingest.state_machine import IngestOrchestrator, IngestResult

# How long to wait after the most recent inbox event before
# triggering an ingest pass. Long enough that `cp -r` of a directory
# of 100 files coalesces into one pass; short enough that the user
# doesn't sit waiting after dropping a single file.
_DEBOUNCE_SECONDS: float = 1.0

_LOG = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    """Internal handler that schedules a debounced callback per inbox event.

    `corpus_root` is the corpus root path; events whose first path
    segment under that root is a reserved nuthatch dir are ignored.
    """

    def __init__(
        self,
        on_quiet: Callable[[], None],
        *,
        corpus_root: Path,
    ) -> None:
        super().__init__()
        self._on_quiet = on_quiet
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._corpus_root = corpus_root

    def _schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(_DEBOUNCE_SECONDS, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        try:
            self._on_quiet()
        except Exception:
            _LOG.exception("ingest pass raised; watcher continuing")

    def _under_reserved_dir(self, src_path: str) -> bool:
        try:
            rel = Path(src_path).resolve().relative_to(self._corpus_root.resolve())
        except ValueError:
            return False
        return bool(rel.parts) and rel.parts[0] in CORPUS_RESERVED_DIRS

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._under_reserved_dir(str(event.src_path)):
            return
        self._schedule()

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # `on_moved` carries dest_path too; if either side is reserved
        # we still want to skip (a move INTO papers/ is the pipeline's
        # own work, a move OUT of papers/ is unusual but not user input).
        if self._under_reserved_dir(str(event.src_path)):
            return
        dest = getattr(event, "dest_path", "")
        if dest and self._under_reserved_dir(str(dest)):
            return
        self._schedule()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class InboxWatcher:
    """Long-running watcher over a corpus root.

    Construct with an `IngestOrchestrator`, call `start()` to begin
    listening, and `stop()` to tear down cleanly. The watcher watches
    the entire `<corpus>/` tree recursively (debounced; reserved
    nuthatch subdirs filtered at the event-handler level). One ingest
    pass also runs immediately on `start()` so files already sitting
    in the corpus at startup are processed without waiting for a new
    event.

    Class name is kept (`InboxWatcher`) for back-compat with existing
    call sites; the scope is now the whole corpus root, not just the
    `inbox/` subdir.
    """

    __slots__ = ("_handler", "_observer", "_on_result", "_orchestrator", "_root")

    def __init__(
        self,
        orchestrator: IngestOrchestrator,
        on_result: Callable[[list[IngestResult]], None] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._root: Path = orchestrator.layout.root
        self._on_result = on_result
        self._handler = _DebouncedHandler(self._run_pass, corpus_root=self._root)
        self._observer = Observer()

    def start(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._observer.schedule(self._handler, str(self._root), recursive=True)
        self._observer.start()
        self._run_pass()  # process anything already sitting in the corpus

    def stop(self) -> None:
        self._handler.cancel()
        self._observer.stop()
        self._observer.join(timeout=5)

    def _run_pass(self) -> None:
        results = self._orchestrator.ingest_corpus()
        if self._on_result is not None and results:
            self._on_result(results)
