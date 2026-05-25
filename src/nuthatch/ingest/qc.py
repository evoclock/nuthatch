# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Quality-control checks for the ingest pipeline.

Purpose: assert pipeline-stage invariants. The Sprint 2 surface is
    `check_extract_yield(markdown)` — confirms an extractor produced
    enough text to be worth keeping. The Sprint 3 surface adds
    `check_chunk_coverage(...)` enforcing the full-document-coverage
    invariant from DECISIONS.md.

Inputs: extractor output (string), thresholds.

Outputs: `CheckResult(passed, reason, details)`.

Assumptions: thresholds are tunable per corpus via config; the
    defaults here are baseline floors derived from the OCR benchmark
    (a paper that extracts under 200 chars total is almost certainly
    broken, not just short).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A real paper extracts at least this many characters total. Even a
# 1-page Nature Letter is ~6 KB of body text. 200 is a generous
# floor that catches truly empty or near-empty extractor output
# without false-positiving on intentionally-short documents.
_MIN_EXTRACT_CHARS: int = 200

# Per-page minimum after the same logic. A scanned PDF where Chandra
# returns < 50 chars/page on average is producing OCR garbage.
_MIN_EXTRACT_CHARS_PER_PAGE: float = 50.0


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str | None
    details: dict[str, Any] = field(default_factory=dict)


def check_extract_yield(
    markdown: str,
    *,
    n_pages: int | None = None,
    min_total: int = _MIN_EXTRACT_CHARS,
    min_per_page: float = _MIN_EXTRACT_CHARS_PER_PAGE,
) -> CheckResult:
    """Did the extractor produce enough text to be worth keeping?

    Both the absolute floor and the per-page floor must pass when
    `n_pages` is supplied.
    """
    total = len(markdown)
    details: dict[str, Any] = {"chars_total": total, "min_total": min_total}
    if total < min_total:
        return CheckResult(
            passed=False,
            reason=f"extract_yield_too_low:{total}<{min_total}",
            details=details,
        )
    if n_pages is not None and n_pages > 0:
        per_page = total / n_pages
        details["chars_per_page"] = per_page
        details["min_per_page"] = min_per_page
        details["n_pages"] = n_pages
        if per_page < min_per_page:
            return CheckResult(
                passed=False,
                reason=(
                    f"extract_yield_per_page_too_low:{per_page:.1f}<{min_per_page}"
                ),
                details=details,
            )
    return CheckResult(passed=True, reason=None, details=details)
