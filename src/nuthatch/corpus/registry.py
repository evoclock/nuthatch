# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Registry of known nuthatch corpora.

Per `docs/DECISIONS.md`, corpora are fully isolated per-directory.
The registry lets the CLI resolve `--corpus <name>` to a filesystem
path without forcing the user to remember the path each invocation.

Default location: `~/.config/nuthatch/registry.toml` (or
`$XDG_CONFIG_HOME/nuthatch/registry.toml` when set).

Format:

    default_corpus = "ml-papers"

    [corpora.ml-papers]
    path = "/home/jgamboa/corpora/ml-papers"

    [corpora.bio-genomics]
    path = "/mnt/encrypted/bio-genomics"

Only `path` is required per entry; future fields (last_used,
created_at, schema_profile) can land additively without breaking
older registry files.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w


def default_registry_path() -> Path:
    """Return the canonical registry path, respecting XDG_CONFIG_HOME."""

    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "nuthatch" / "registry.toml"


@dataclass(slots=True)
class _CorpusEntry:
    """Per-corpus registry entry."""

    path: Path


@dataclass(slots=True)
class Registry:
    """In-memory view of the nuthatch corpora registry.

    Backed by a TOML file on disk. `load` reads from a path (the
    default is `default_registry_path()`); `save` writes the
    current in-memory state back atomically. `add`, `remove`, and
    `set_default` mutate the in-memory state — call `save`
    explicitly to persist.
    """

    corpora: dict[str, _CorpusEntry] = field(default_factory=dict)
    default_corpus: str | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> Registry:
        """Read the registry from disk; return empty Registry on missing file."""

        resolved = path if path is not None else default_registry_path()
        if not resolved.is_file():
            return cls()

        raw: dict[str, Any] = tomllib.loads(resolved.read_text(encoding="utf-8"))
        default = raw.get("default_corpus")
        corpora_raw = raw.get("corpora", {}) or {}
        corpora: dict[str, _CorpusEntry] = {}
        for name, entry in corpora_raw.items():
            if not isinstance(entry, dict):
                continue
            entry_path = entry.get("path")
            if not isinstance(entry_path, str) or not entry_path:
                continue
            corpora[name] = _CorpusEntry(path=Path(entry_path))

        return cls(
            corpora=corpora,
            default_corpus=default if isinstance(default, str) else None,
        )

    def save(self, path: Path | None = None) -> None:
        """Write the registry to disk atomically."""

        resolved = path if path is not None else default_registry_path()
        resolved.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "corpora": {
                name: {"path": str(entry.path)} for name, entry in sorted(self.corpora.items())
            },
        }
        if self.default_corpus is not None:
            payload = {"default_corpus": self.default_corpus, **payload}

        tmp = resolved.with_suffix(resolved.suffix + ".tmp")
        tmp.write_text(tomli_w.dumps(payload), encoding="utf-8")
        tmp.replace(resolved)

    def add(self, name: str, path: Path) -> None:
        """Register a corpus. Overwrites any existing entry with the same name."""
        self.corpora[name] = _CorpusEntry(path=path.resolve())

    def remove(self, name: str) -> bool:
        """Unregister a corpus by name. Returns True if removed; False if absent."""
        if name not in self.corpora:
            return False
        del self.corpora[name]
        if self.default_corpus == name:
            self.default_corpus = None
        return True

    def set_default(self, name: str) -> None:
        """Mark `name` as the registry's default corpus."""
        if name not in self.corpora:
            raise KeyError(f"unknown corpus: {name!r}")
        self.default_corpus = name

    def resolve(self, name: str | None) -> Path | None:
        """Return the path for `name`, or for the default if `name` is None."""
        target = name if name is not None else self.default_corpus
        if target is None:
            return None
        entry = self.corpora.get(target)
        return entry.path if entry is not None else None

    def names(self) -> list[str]:
        return sorted(self.corpora.keys())
