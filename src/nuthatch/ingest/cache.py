# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Semantic cache for the ingest pipeline.

Purpose: skip re-extraction (the expensive OCR step) when the same
    file with the same content + same extractor version has been
    processed before. Cache key = `(content_hash, extractor_version,
    profile_name)`; cache value = extracted markdown plus its
    associated metadata.

Inputs: a corpus cache directory (`<corpus>/.kg/cache/`) and the
    triplet that identifies a reusable extract.

Outputs: cached markdown / metadata when hit; `None` on miss.

Pattern reused from `a prior implementation`'s `cache.py` (the semantic cache that
short-circuits repeat work in the external pipeline library
nuthatch borrows from). nuthatch's implementation is a fresh write
shaped by that pattern: a per-corpus on-disk cache keyed by the
file content hash plus a per-stage signature, with markdown stored
in a flat directory and metadata stored alongside as JSON sidecars.

Assumptions: cache lives under the corpus, not globally — different
    corpora may use different schema profiles or extractor versions
    and should not share extracts. A change to extractor version
    invalidates only the entries with the old version (others stay
    usable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    """Triplet that identifies a reusable extracted artifact."""

    content_hash: str
    extractor_version: str
    profile_name: str

    def stem(self) -> str:
        """Filesystem-safe stem for the cache entry filename."""
        # 12 hash chars are enough to avoid collisions for a per-corpus
        # cache; full hash is in the sidecar JSON.
        short = self.content_hash[:12]
        ev = self.extractor_version.replace("/", "_")
        pn = self.profile_name.replace("/", "_")
        return f"{short}__{ev}__{pn}"


class SemanticCache:
    """Per-corpus on-disk cache mapping `CacheKey` to extracted markdown.

    Storage layout under `<root>/`:

    - `<stem>.md` — the cached extracted markdown
    - `<stem>.meta.json` — the metadata sidecar (full hash, extractor
      version, profile name, timestamp, length)

    Used by the orchestrator: on inbox ingest, hash the file, build
    the `CacheKey`, ask the cache. On hit, skip extract + qc, run
    schema validation against the cached metadata directly.
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: CacheKey, suffix: str) -> Path:
        return self._root / f"{key.stem()}{suffix}"

    def has(self, key: CacheKey) -> bool:
        return self._path(key, ".md").is_file()

    def get(self, key: CacheKey) -> tuple[str, dict[str, Any]] | None:
        """Return `(markdown, metadata)` on hit; `None` on miss."""
        md_path = self._path(key, ".md")
        meta_path = self._path(key, ".meta.json")
        if not md_path.is_file():
            return None
        markdown = md_path.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        return markdown, metadata

    def put(
        self,
        key: CacheKey,
        *,
        markdown: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store `markdown` and optional metadata under `key`."""
        self._root.mkdir(parents=True, exist_ok=True)
        self._path(key, ".md").write_text(markdown, encoding="utf-8")
        sidecar = {
            "content_hash": key.content_hash,
            "extractor_version": key.extractor_version,
            "profile_name": key.profile_name,
            "chars": len(markdown),
        }
        if metadata:
            sidecar["metadata"] = metadata
        self._path(key, ".meta.json").write_text(
            json.dumps(sidecar, indent=2, default=str), encoding="utf-8"
        )

    def invalidate(self, key: CacheKey) -> None:
        for suffix in (".md", ".meta.json"):
            p = self._path(key, suffix)
            if p.exists():
                p.unlink()

    def invalidate_extractor_version(self, extractor_version: str) -> int:
        """Drop every entry written under a given extractor version. Returns count."""
        if not self._root.is_dir():
            return 0
        n = 0
        marker = f"__{extractor_version.replace('/', '_')}__"
        for p in list(self._root.iterdir()):
            if marker in p.name:
                p.unlink()
                if p.name.endswith(".md"):
                    n += 1
        return n
