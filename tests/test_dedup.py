# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for `nuthatch.ingest.dedup`."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nuthatch.ingest.dedup import hash_file


class TestHashFile:
    def test_hash_matches_hashlib_for_small_file(self, tmp_path: Path) -> None:
        path = tmp_path / "small.txt"
        path.write_bytes(b"hello nuthatch")
        expected = hashlib.sha256(b"hello nuthatch").hexdigest()
        assert hash_file(path) == expected

    def test_hash_matches_for_large_file(self, tmp_path: Path) -> None:
        # 256 KiB of random-ish bytes to exercise the streaming reader.
        path = tmp_path / "large.bin"
        payload = bytes(range(256)) * 1024
        path.write_bytes(payload)
        assert hash_file(path) == hashlib.sha256(payload).hexdigest()

    def test_hash_is_stable_across_calls(self, tmp_path: Path) -> None:
        path = tmp_path / "stable.txt"
        path.write_bytes(b"stable content")
        assert hash_file(path) == hash_file(path)

    def test_distinct_content_distinct_hashes(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.write_bytes(b"first")
        b.write_bytes(b"second")
        assert hash_file(a) != hash_file(b)

    def test_raises_filenotfound_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            hash_file(tmp_path / "does-not-exist.pdf")
