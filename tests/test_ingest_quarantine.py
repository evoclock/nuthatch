# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.ingest.quarantine.quarantine_file`."""

from __future__ import annotations

import json
from pathlib import Path

from nuthatch.ingest.quarantine import quarantine_file


def _make_source(tmp_path: Path, name: str = "p.pdf", body: bytes = b"x") -> Path:
    inbox = tmp_path / "inbox"
    inbox.mkdir(exist_ok=True)
    p = inbox / name
    p.write_bytes(body)
    return p


class TestQuarantineFile:
    def test_moves_file_to_reason_subdir(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        qroot = tmp_path / "q"
        dest = quarantine_file(source, qroot, reason="unsupported_format:.pdf")
        assert dest.exists()
        assert not source.exists()
        # Reason got slugified into a directory name.
        assert dest.parent.parent == qroot
        assert dest.parent.name  # non-empty slug

    def test_writes_reason_sidecar_json(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        dest = quarantine_file(
            source,
            tmp_path / "q",
            reason="missing:title,year",
            details={"profile": "arxiv_paper", "found": ["abstract"]},
        )
        sidecar = dest.with_suffix(dest.suffix + ".reason.json")
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text())
        assert payload["reason"] == "missing:title,year"
        assert payload["original_filename"] == "p.pdf"
        assert payload["details"] == {"profile": "arxiv_paper", "found": ["abstract"]}
        assert "quarantined_at_utc" in payload

    def test_slug_sanitises_unsafe_chars(self, tmp_path: Path) -> None:
        source = _make_source(tmp_path)
        dest = quarantine_file(source, tmp_path / "q", reason="weird / reason: with spaces!")
        # Slug should be filesystem-safe.
        assert "/" not in dest.parent.name
        assert " " not in dest.parent.name
        assert "!" not in dest.parent.name

    def test_collision_appends_counter(self, tmp_path: Path) -> None:
        qroot = tmp_path / "q"
        first = _make_source(tmp_path, name="p.pdf", body=b"first")
        d1 = quarantine_file(first, qroot, reason="same_reason")
        second = _make_source(tmp_path, name="p.pdf", body=b"second")
        d2 = quarantine_file(second, qroot, reason="same_reason")
        assert d1.name == "p.pdf"
        assert d2.name == "p-1.pdf"
        assert d1.read_bytes() == b"first"
        assert d2.read_bytes() == b"second"
