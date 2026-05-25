# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for `nuthatch.ingest.extract`.

Scope: pure-logic tests against `pick_strategy` (no real PDFs, no
GPU). Real-extraction integration tests live elsewhere; they exercise
heavy backends and are slow.
"""

from __future__ import annotations

import pytest

from nuthatch.ingest.extract import ExtractionStrategy, pick_strategy


class TestPickStrategy:
    def test_high_text_yield_routes_to_docling(self) -> None:
        assert (
            pick_strategy(4000.0, has_gpu=True)
            is ExtractionStrategy.DIGITAL_DOCLING
        )

    def test_high_text_yield_no_gpu_still_docling(self) -> None:
        assert (
            pick_strategy(4000.0, has_gpu=False)
            is ExtractionStrategy.DIGITAL_DOCLING
        )

    def test_low_yield_with_gpu_max_quality_routes_to_chandra(self) -> None:
        assert (
            pick_strategy(50.0, has_gpu=True, prefer_max_quality=True)
            is ExtractionStrategy.SCANNED_CHANDRA
        )

    def test_low_yield_with_gpu_smaller_model_routes_to_granite(self) -> None:
        assert (
            pick_strategy(50.0, has_gpu=True, prefer_max_quality=False)
            is ExtractionStrategy.SCANNED_GRANITE
        )

    def test_low_yield_no_gpu_routes_to_easyocr(self) -> None:
        assert (
            pick_strategy(50.0, has_gpu=False)
            is ExtractionStrategy.SCANNED_EASYOCR
        )


class TestThresholdBoundary:
    @pytest.mark.parametrize(
        "yield_value,expected",
        [
            (2999.9, ExtractionStrategy.SCANNED_CHANDRA),
            (3000.0, ExtractionStrategy.DIGITAL_DOCLING),
            (3000.1, ExtractionStrategy.DIGITAL_DOCLING),
            (0.0, ExtractionStrategy.SCANNED_CHANDRA),
            (1660.0, ExtractionStrategy.SCANNED_CHANDRA),  # Mendel-class
            (2382.0, ExtractionStrategy.SCANNED_CHANDRA),  # Wright-class
            (4000.0, ExtractionStrategy.DIGITAL_DOCLING),  # modern preprint floor
        ],
    )
    def test_threshold_at_3000_chars_per_page(
        self, yield_value: float, expected: ExtractionStrategy
    ) -> None:
        assert pick_strategy(yield_value, has_gpu=True) is expected

    def test_extreme_high_yield(self) -> None:
        assert (
            pick_strategy(1_000_000.0, has_gpu=True)
            is ExtractionStrategy.DIGITAL_DOCLING
        )
