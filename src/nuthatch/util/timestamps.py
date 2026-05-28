# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Shared timestamp helpers for output filenames.

Eliminates 6 copies of `_now_tag` previously duplicated across the
eval, concepts, and viz scripts. Single source so renames or format
changes happen in one place.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_tag() -> str:
    """Return a UTC timestamp suitable for a filename: `YYYYMMDDTHHMMSSZ`."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
