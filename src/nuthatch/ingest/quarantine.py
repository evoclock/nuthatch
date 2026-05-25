# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Quarantine routing for files that fail the ingest gate.

Purpose: move a failed-ingest file to `<corpus>/quarantine/<reason>/`
    and write a `.reason.json` sidecar capturing why. Per the
    DECISIONS.md ingest rule, schema-failed files do NOT enter the
    graph and must be inspectable by hand later.

Inputs: source path, corpus quarantine root, reason string,
    optional details dict (extracted metadata, missing fields, raw
    OCR confidence, etc.).

Outputs: the final quarantine path. A sidecar `<file>.reason.json`
    lands next to it.

Assumptions: caller has already decided this file should quarantine;
    this module does not re-validate. Reason strings should be short
    snake_case slugs (e.g. `missing_required_fields`,
    `unsupported_format`, `extract_empty`) so they form usable
    subdirectory names.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reason slugs become directory names; sanitize aggressively.
_SLUG_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(reason: str) -> str:
    """Reason → safe directory-name slug. Collapses bad chars to `_`."""
    cleaned = _SLUG_SAFE_RE.sub("_", reason.strip()).strip("_")
    return cleaned or "unknown"


def quarantine_file(
    source: Path,
    quarantine_root: Path,
    *,
    reason: str,
    details: dict[str, Any] | None = None,
) -> Path:
    """Move `source` to `<quarantine_root>/<reason-slug>/` + write sidecar.

    Returns the final destination path of the moved file.
    """
    reason_slug = _slug(reason)
    target_dir = quarantine_root / reason_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    dest = target_dir / source.name
    # Resolve filename collision by appending -1, -2, ...
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        counter = 1
        while dest.exists():
            dest = target_dir / f"{stem}-{counter}{suffix}"
            counter += 1

    shutil.move(str(source), str(dest))

    sidecar = dest.with_suffix(dest.suffix + ".reason.json")
    payload = {
        "reason": reason,
        "reason_slug": reason_slug,
        "original_filename": source.name,
        "quarantined_at_utc": datetime.now(UTC).isoformat(),
        "details": details or {},
    }
    sidecar.write_text(json.dumps(payload, indent=2, default=str))
    return dest
