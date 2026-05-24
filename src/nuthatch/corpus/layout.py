# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Per-corpus directory layout.

A nuthatch corpus is a self-contained directory. The `.kg/` subdir
is the marker (analogous to `.git/`); finding it by walking up the
tree is how the CLI resolves "the current corpus" when the user
doesn't pass `--corpus` explicitly.

Layout (per `docs/SPEC.md`):

    my-corpus/
    ├── .kg/                  tool bookkeeping (manifest, config, audit)
    ├── inbox/                drop files here; watcher / `nuthatch ingest` picks up
    ├── quarantine/           schema-failed; awaiting metadata fix
    ├── papers/               original sources (PDFs, HTML, etc.)
    ├── cards/                MD card per paper (Sprint 3)
    ├── html/                 HTML companion per paper (Sprint 3)
    ├── notes/                user-imported notes
    ├── graph/                graph state + SBM block state (Sprint 4)
    └── exports/              generated outputs (Sprint 6)

`init_corpus` creates the full tree on first run. `CorpusLayout`
exposes the canonical paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Sentinel directory name that marks "this is a nuthatch corpus
# root". Discovery walks up the directory tree looking for it.
CORPUS_MARKER: str = ".kg"

# Subdirectories created by `init_corpus`. Each one is created with
# a `.gitkeep` so the structure is preserved when corpora are
# version-controlled with the contents gitignored.
_STANDARD_SUBDIRS: tuple[str, ...] = (
    "inbox",
    "quarantine",
    "papers",
    "cards",
    "html",
    "notes",
    "graph",
    "exports",
)


@dataclass(frozen=True, slots=True)
class CorpusLayout:
    """Resolved paths for a single corpus.

    Construct via `init_corpus` (for a new corpus) or
    `discover_corpus_root` + `CorpusLayout(root)` (for an existing
    one). Paths are absolute and the existence of the corpus root
    is the only invariant; child subdirs may be missing if the
    corpus is being walked through migration.
    """

    root: Path

    @property
    def kg(self) -> Path:
        return self.root / CORPUS_MARKER

    @property
    def manifest_path(self) -> Path:
        return self.kg / "manifest.jsonl"

    @property
    def config_path(self) -> Path:
        return self.kg / "config.yaml"

    @property
    def audit_dir(self) -> Path:
        return self.kg / "audit"

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def quarantine(self) -> Path:
        return self.root / "quarantine"

    @property
    def papers(self) -> Path:
        return self.root / "papers"

    @property
    def cards(self) -> Path:
        return self.root / "cards"

    @property
    def html(self) -> Path:
        return self.root / "html"

    @property
    def notes(self) -> Path:
        return self.root / "notes"

    @property
    def graph(self) -> Path:
        return self.root / "graph"

    @property
    def exports(self) -> Path:
        return self.root / "exports"


def init_corpus(root: Path) -> CorpusLayout:
    """Create the `.kg/` marker + the standard subdir tree at `root`.

    Idempotent: re-running against an existing corpus is a no-op.
    Returns the resolved `CorpusLayout` whether the corpus was
    freshly created or pre-existed.
    """

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    layout = CorpusLayout(root=root)

    layout.kg.mkdir(parents=True, exist_ok=True)
    layout.audit_dir.mkdir(parents=True, exist_ok=True)
    for name in _STANDARD_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)

    return layout


def discover_corpus_root(start: Path) -> Path | None:
    """Walk up the directory tree looking for a `.kg/` marker.

    Returns the corpus root (the directory containing `.kg/`), or
    `None` if no corpus is found before hitting `/` or the user's
    home directory.

    Bounded at home so a stray invocation in `/tmp` (or wherever)
    doesn't traverse the entire filesystem looking for a marker
    that doesn't exist.
    """

    start = start.resolve()
    home = Path.home().resolve()
    current = start if start.is_dir() else start.parent

    while True:
        if (current / CORPUS_MARKER).is_dir():
            return current
        if current == current.parent:
            return None  # reached filesystem root
        if current == home:
            return None  # don't search above the user's home
        current = current.parent
