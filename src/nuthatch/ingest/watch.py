# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Filesystem-event-driven ingest for `nuthatch watch`.

Uses `watchdog` to listen for file-create / file-move events under
`<corpus>/inbox/`. Each event schedules a debounced re-run of
`IngestOrchestrator.ingest_inbox()` (debouncing matters because
batch copy operations fire one event per file; we want to handle
them as a single ingest pass instead of N separate ones).

This is the long-running mode. One-shot ingest stays available via
`IngestOrchestrator.ingest_inbox()` invoked from the CLI's
`nuthatch ingest` subcommand. Watch mode is `nuthatch watch`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from nuthatch.ingest.state_machine import IngestOrchestrator, IngestResult

# How long to wait after the most recent inbox event before
# triggering an ingest pass. Long enough that `cp -r` of a directory
# of 100 files coalesces into one pass; short enough that the user
# doesn't sit waiting after dropping a single file.
_DEBOUNCE_SECONDS: float = 1.0

_LOG = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    """Internal handler that schedules a debounced callback per inbox event."""

    def __init__(self, on_quiet: Callable[[], None]) -> None:
        super().__init__()
        self._on_quiet = on_quiet
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

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

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule()

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class InboxWatcher:
    """Long-running watcher over a corpus's inbox directory.

    Construct with an `IngestOrchestrator`, call `start()` to begin
    listening, and `stop()` to tear down cleanly. The watcher also
    runs one ingest pass immediately on `start()` so files already
    sitting in the inbox at startup are processed without waiting
    for a new event.
    """

    __slots__ = ("_handler", "_inbox", "_observer", "_on_result", "_orchestrator")

    def __init__(
        self,
        orchestrator: IngestOrchestrator,
        on_result: Callable[[list[IngestResult]], None] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._inbox: Path = orchestrator.layout.inbox
        self._on_result = on_result
        self._handler = _DebouncedHandler(self._run_pass)
        self._observer = Observer()

    def start(self) -> None:
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._observer.schedule(self._handler, str(self._inbox), recursive=True)
        self._observer.start()
        self._run_pass()  # process anything already sitting in the inbox

    def stop(self) -> None:
        self._handler.cancel()
        self._observer.stop()
        self._observer.join(timeout=5)

    def _run_pass(self) -> None:
        results = self._orchestrator.ingest_inbox()
        if self._on_result is not None and results:
            self._on_result(results)
