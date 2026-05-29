# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for `nuthatch.ingest.cache.SemanticCache`."""

from __future__ import annotations

import json
from pathlib import Path

from nuthatch.ingest.cache import CacheKey, SemanticCache


def _key(content_hash: str = "abc123def456" * 6) -> CacheKey:
    return CacheKey(
        content_hash=content_hash,
        extractor_version="router-v1",
        profile_name="arxiv_paper",
    )


class TestSemanticCache:
    def test_miss_on_empty(self, tmp_path: Path) -> None:
        cache = SemanticCache(tmp_path / "cache")
        assert cache.has(_key()) is False
        assert cache.get(_key()) is None

    def test_round_trip(self, tmp_path: Path) -> None:
        cache = SemanticCache(tmp_path / "cache")
        md = "# Title\n\nLorem ipsum body."
        meta = {"title": "Title", "year": 2024}
        cache.put(_key(), markdown=md, metadata=meta)
        assert cache.has(_key())
        result = cache.get(_key())
        assert result is not None
        markdown, sidecar = result
        assert markdown == md
        assert sidecar["metadata"] == meta
        assert sidecar["chars"] == len(md)
        assert sidecar["extractor_version"] == "router-v1"

    def test_different_extractor_version_misses(self, tmp_path: Path) -> None:
        cache = SemanticCache(tmp_path / "cache")
        cache.put(_key(), markdown="A", metadata={})
        other = CacheKey(
            content_hash=_key().content_hash,
            extractor_version="router-v2",
            profile_name="arxiv_paper",
        )
        assert cache.get(other) is None

    def test_invalidate(self, tmp_path: Path) -> None:
        cache = SemanticCache(tmp_path / "cache")
        cache.put(_key(), markdown="A", metadata={})
        cache.invalidate(_key())
        assert cache.get(_key()) is None

    def test_invalidate_extractor_version_bulk(self, tmp_path: Path) -> None:
        cache = SemanticCache(tmp_path / "cache")
        for h in ("aa" * 8, "bb" * 8, "cc" * 8):
            cache.put(
                CacheKey(content_hash=h, extractor_version="v1", profile_name="p"),
                markdown="x",
                metadata={},
            )
        cache.put(
            CacheKey(content_hash="dd" * 8, extractor_version="v2", profile_name="p"),
            markdown="y",
            metadata={},
        )
        dropped = cache.invalidate_extractor_version("v1")
        assert dropped == 3
        # v2 entry survives.
        assert (
            cache.get(CacheKey(content_hash="dd" * 8, extractor_version="v2", profile_name="p"))
            is not None
        )

    def test_sidecar_is_valid_json(self, tmp_path: Path) -> None:
        cache = SemanticCache(tmp_path / "cache")
        cache.put(_key(), markdown="md", metadata={"k": "v"})
        sidecar = (tmp_path / "cache" / f"{_key().stem()}.meta.json").read_text()
        parsed = json.loads(sidecar)
        assert parsed["profile_name"] == "arxiv_paper"
