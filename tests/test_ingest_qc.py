# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.ingest.qc.check_extract_yield`."""

from __future__ import annotations

from nuthatch.ingest.qc import check_extract_yield


class TestCheckExtractYield:
    def test_long_text_passes(self) -> None:
        r = check_extract_yield("x" * 5000)
        assert r.passed
        assert r.reason is None

    def test_short_text_fails(self) -> None:
        r = check_extract_yield("x" * 50)
        assert not r.passed
        assert r.reason is not None
        assert "extract_yield_too_low" in r.reason

    def test_empty_text_fails(self) -> None:
        r = check_extract_yield("")
        assert not r.passed

    def test_per_page_floor_independent_of_total(self) -> None:
        # Total passes (1500 chars) but per-page fails (1500 / 50 pages = 30/p).
        r = check_extract_yield("x" * 1500, n_pages=50)
        assert not r.passed
        assert r.reason is not None
        assert "per_page_too_low" in r.reason

    def test_both_floors_satisfied(self) -> None:
        # 3000 chars over 10 pages = 300/p; both pass.
        r = check_extract_yield("x" * 3000, n_pages=10)
        assert r.passed
        assert r.details["chars_per_page"] == 300

    def test_custom_thresholds(self) -> None:
        # Customise floor so a tiny doc passes.
        r = check_extract_yield("hi", min_total=1)
        assert r.passed
