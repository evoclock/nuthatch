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
    ├── <user-subdirs>/       arxiv/, bioarxiv/, notes/, benchmark_test/, ...
    ├── processed/            successfully ingested sources, mirroring the user's
    │   ├── arxiv/            origin subdir (provenance preserved). The watcher
    │   ├── bioarxiv/         is fed from inputs/ only, not from here.
    │   └── ...
    ├── quarantine/           transient state: recoverable failure (schema, qc).
    │   └── <reason>/         A fix-pass either resolves to processed/<orig_subdir>/
    │                         or escalates to rejected/<reason>/.
    ├── rejected/             terminal state: declared unfixable post-investigation
    ├── cards/                MD card per paper (Sprint 3)
    ├── html/                 HTML companion per paper (Sprint 3)
    ├── graph/                graph state + SBM block state (Sprint 4)
    └── exports/              generated outputs (Sprint 6)

Lifecycle: `inputs/<subdir>/foo.pdf` -> success -> `processed/<subdir>/foo.pdf`.
On schema/QC failure -> `quarantine/<reason>/foo.pdf` with a sidecar
recording `original_subdir`. Quarantine is transient; the only legal
exits are processed/<original_subdir>/ (when fixed) or rejected/<reason>/
(when declared unfixable). Files never leave the corpus.

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
    "processed",
    "quarantine",
    "rejected",
    "cards",
    "html",
    "notes",
    "graph",
    "exports",
)

# Reserved subdirectory names that hold nuthatch-managed artifacts
# (derived state, rendered output, or originals already moved by a
# previous ingest pass). The recursive corpus scan in the ingest
# orchestrator (and the InboxWatcher) skip these trees entirely.
#
# Crucially, this list does NOT include `inbox/` or `notes/`: those
# are user-source dirs that should still be scanned. Files under any
# other subdir the user creates (e.g. `arxiv/`, `bioarxiv/`,
# `papers/2026/`) are picked up by the recursive scan.
CORPUS_RESERVED_DIRS: frozenset[str] = frozenset(
    {
        CORPUS_MARKER,  # ".kg"
        "processed",  # post-ingest originals (provenance preserved by subdir)
        "quarantine",  # transient failures (awaiting fix or rejection)
        "rejected",  # terminal failures (declared unfixable)
        "cards",  # rendered markdown
        "html",  # rendered HTML companion
        "communities",  # rendered community pages
        "graph",  # derived graph state
        "exports",  # rendered outputs
        "reports",  # generated reports (token-econ, decay)
        "defer",  # papers deferred from ingest (e.g. post-Thursday
        # triage); reserved exactly like benchmark_test/
        "benchmark_test",  # user-curated reference PDFs for OCR / extraction
        # benchmarks; not part of the queryable corpus and
        # not auto-scanned. Conventional name; users with a
        # different convention can opt out via the per-corpus
        # `scan_skip` config (see `corpus/config.py`).
    }
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
    def extracted_dir(self) -> Path:
        """`<corpus>/.kg/extracted/` — markdown + meta per ingested doc.

        Populated by `IngestOrchestrator` on successful schema-validation.
        Consumed by `nuthatch embed` (chunker reads the markdown; meta
        carries title/year/topics for per-chunk metadata in Chroma).

        Lives under `.kg/` (not in `papers/`) because it's a derived
        artifact: re-extraction would reproduce it, so it's gitignored
        as part of the `.kg/` tree.
        """
        return self.kg / "extracted"

    @property
    def embeddings_dir(self) -> Path:
        """`<corpus>/.kg/embeddings/` — Chroma persistent store root."""
        return self.kg / "embeddings"

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def quarantine(self) -> Path:
        return self.root / "quarantine"

    @property
    def processed(self) -> Path:
        """`<corpus>/processed/` — successfully-ingested sources.

        Mirrors the source's relative path under `<corpus>/`. A file
        ingested from `<corpus>/bioarxiv/foo.pdf` lands at
        `<corpus>/processed/bioarxiv/foo.pdf`. The watcher / scan
        skips this subtree so processed files are not re-fed.
        """
        return self.root / "processed"

    @property
    def rejected(self) -> Path:
        """`<corpus>/rejected/` — files declared unfixable post-investigation.

        Files only land here as an explicit decision by the operator
        (or a fix-pass), never directly from the ingest pipeline. The
        ingest pipeline's two failure outcomes are `quarantine/<reason>/`
        (recoverable) and infrastructure-FAILED (transient, no move).
        """
        return self.root / "rejected"

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

    def iter_source_files(self) -> list[Path]:
        """Sorted list of real files anywhere under the corpus root.

        Walks the entire corpus tree, skipping `CORPUS_RESERVED_DIRS`
        (nuthatch-managed artifact dirs) and hidden files. Lets users
        organise their inputs however they like (`arxiv/2026/`,
        `inbox/`, `notes/`, root-level PDFs, etc.) without the ingest
        layer forcing a flat `inbox/` structure.
        """
        files: list[Path] = []
        for path in self.root.rglob("*"):
            if path.is_dir():
                continue
            # Skip any file whose path goes through a reserved dir.
            try:
                rel_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if rel_parts and rel_parts[0] in CORPUS_RESERVED_DIRS:
                continue
            if path.name.startswith("."):
                continue
            files.append(path)
        files.sort()
        return files


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
    layout.extracted_dir.mkdir(parents=True, exist_ok=True)
    layout.embeddings_dir.mkdir(parents=True, exist_ok=True)
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
