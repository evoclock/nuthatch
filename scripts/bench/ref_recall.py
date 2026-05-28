#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""
Script: ref_recall

Path: scripts/bench/ref_recall.py

Purpose: count distinct numbered reference entries in an OCR markdown
    output. Used by the extraction benchmark in
    docs/extraction-benchmarks/ to quantify which OCR tool extracted
    the most-recoverable bibliography.

Inputs: one or more paths to markdown files produced by an OCR
    extraction tool.

Outputs: stdout, one line per file: `<count>\\t<filename>`.

Assumptions: references are numbered with a leading integer followed
    by a period (`8. Author, ...` or `8 . Author, ...`). Some OCR
    tools emit the count as bullet-list items (`- 8.`); the regex
    accepts that form too.

Parameters: none. Threshold and regex are constants.

Failure Modes: a tool that hallucinates extra numeric tokens at the
    start of lines will over-count. The metric is a recall proxy,
    not a precision measure.

Author: Julen Gamboa

Created: 2026-05-25

Last Edited: 2026-05-25 by Julen Gamboa
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Match lines that start a numbered reference entry. Allows optional
# leading bullet marker and whitespace before the integer + period.
_REF_LINE_RE = re.compile(r"^[-*]?\s*(\d{1,3})\.\s+\S", re.MULTILINE)


def count_distinct_refs(markdown: str) -> int:
    """Count distinct reference numbers that appear as line starters."""
    numbers = {int(m.group(1)) for m in _REF_LINE_RE.finditer(markdown)}
    return len(numbers)


def main(paths: list[str]) -> int:
    if not paths:
        print("usage: ref_recall.py <file.md> [<file.md> ...]", file=sys.stderr)
        return 2
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        count = count_distinct_refs(text)
        print(f"{count}\t{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
