# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Resolve a corpus identifier to its root directory.

Single source of truth for the `--corpus NAME_OR_PATH` flag that every
CLI subcommand accepts. Tries the registry first (named corpus), then
the filesystem (a path containing `.kg/`), then `discover_corpus_root`
(walk up from the given path looking for the marker). Raises
`SystemExit` with a friendly message when nothing resolves so CLI
subcommands can fail fast.

Eliminates 8 copies of `_resolve_corpus` previously duplicated across
scripts/eval/, scripts/concepts/, scripts/viz/.
"""

from __future__ import annotations

from pathlib import Path

from nuthatch.corpus import Registry, discover_corpus_root


def resolve_corpus(name_or_path: str) -> Path:
    """Return the resolved corpus root directory for a name or path."""
    try:
        registry = Registry.load()
        entry = registry.resolve(name_or_path)
        if entry is not None:
            return Path(entry).resolve()
    except Exception:
        # Registry load can fail for many reasons (no registry,
        # corrupt file, etc.); fall through to filesystem checks.
        pass
    p = Path(name_or_path).resolve()
    if (p / ".kg").is_dir():
        return p
    # Walk up from the given path looking for a `.kg/` marker.
    found = discover_corpus_root(p)
    if found is not None:
        return found
    raise SystemExit(f"corpus not found: {name_or_path!r}")
