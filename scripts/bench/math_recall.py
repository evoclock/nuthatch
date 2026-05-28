#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""
Script: math_recall

Path: scripts/bench/math_recall.py

Purpose: count math notation an OCR tool preserved in its output. The
    key_facts checklist probes textual facts and misses what
    differentiates OCR tools on math-heavy papers (equation fidelity,
    LaTeX rendering, sub/superscript preservation). This metric
    closes that gap.

Inputs: one or more markdown files produced by an OCR extraction.
    Optional `--paper` selects a per-paper known-equations
    checklist for targeted recall scoring.

Outputs: stdout per file, with three counters and (when `--paper` is
    supplied) a per-equation checklist:

    inline_math:     <count>   # `$...$` spans
    block_math:      <count>   # `$$...$$` or fenced math blocks
    math_symbols:    <count>   # math operators, Greek letters, LaTeX commands
    [ok|miss]  equation: <label>
    ...
    equation_recall: N/M

Assumptions: tools that emit no LaTeX (`$...$` etc.) score zero on
    inline / block math but may still match the equation checklist
    via plain-text equivalents (e.g. `G = 7.43`). Inline math count
    and equation recall are independent signals.

Author: Julen Gamboa

Created: 2026-05-25

Last Edited: 2026-05-25 by Julen Gamboa
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Inline math: `$...$` not preceded by `$` and not the start of `$$...$$`.
# Block math: `$$...$$` greedy across lines, OR triple-backtick math blocks.
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^$\n]{1,200}?)\$(?!\$)")
_BLOCK_MATH_RE = re.compile(r"\$\$.+?\$\$", re.DOTALL)

# Math symbols + Greek letters + common LaTeX commands. Each match
# counts once toward `math_symbols`.
_MATH_SYMBOLS_RE = re.compile(
    r"[ΔΣ∫≤≥≈∂∇∞"
    r"αβγδεζηθικλμ"
    r"νξοπρστυφχψω]"
    r"|\\(?:frac|sum|int|alpha|beta|gamma|delta|epsilon|zeta|theta|lambda|mu|"
    r"nu|pi|sigma|tau|phi|omega|leq|geq|approx|partial|nabla|infty|sqrt|cdot|"
    r"times|to|prod|log|exp|sin|cos|tan)"
)


def _count(pattern: re.Pattern[str], text: str) -> int:
    return sum(1 for _ in pattern.finditer(text))


# Per-paper checklists of known equations / formulae. Match strings are
# substring-anchored after whitespace collapse; case-insensitive.
EQUATION_CHECKLISTS: dict[str, list[tuple[str, str]]] = {
    "mk1991": [
        ("G-test statistic G=7.43", r"G\s*=\s*7\.43"),
        ("p-value P=0.006", r"P\s*=\s*0\.00[56]"),
        ("Replacement fixed = 7", r"Replacement[^\n]{0,80}7|7[^\n]{0,80}Replacement"),
        ("Synonymous fixed = 17", r"Synonymous[^\n]{0,80}17|17[^\n]{0,80}Synonymous"),
    ],
    "wright1931": [
        ("rate of loss 1/2N", r"1\s*/\s*2\s*\$?\s*N\s*\$?"),
        ("selection ratio (1-s):1", r"\(\s*1\s*[-−]\s*s\s*\)\s*:\s*1"),  # noqa: RUF001
        ("change in q (Δq formula)", r"\\Delta\s*q|Δ\s*q"),
        ("gene-array expression with q and A", r"q\s*A|\(\s*1\s*[-−]\s*q\s*\)\s*a"),  # noqa: RUF001
        ("subscript N_m (male population size)", r"N\s*[_]?\s*m\b|N\$?_\{?m\}?"),
        ("squared term q^2 / q\\^2", r"q\s*\^?\s*2|q\^2|q\$\^2"),
    ],
    "mendel": [
        ("3:1 ratio (numeric form)", r"\b3\s*:\s*1\b"),
        (
            "phenotype expression 1A:2Aa:1a-ish",
            r"\b1\s*[A-Za-z]?\s*:\s*2\s*[A-Za-z]{1,3}\s*:\s*1\s*[A-Za-z]?\b",
        ),
    ],
}


def _normalise_for_equations(text: str) -> str:
    """Strip markdown emphasis + HTML tags for equation pattern matching."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans({"*": " ", "_": " ", "`": " "}))
    return re.sub(r"\s+", " ", text)


def _classify_inline_math(text: str) -> tuple[int, int]:
    """Return (real_equations, broken_fragments) inside inline-math spans.

    A real equation has either an operator (`=`, `+`, `-`, `*`, `/`,
    `^`, `<`, `>`) or contains more than one alphabetic variable. A
    fragment is something like `\\_{s}`, `^{6}`, or just a stray
    `\\mu` — common Docling-VLM failure modes where LaTeX subscripts
    or superscripts get emitted without the host variable.
    """
    real = 0
    broken = 0
    operator_re = re.compile(r"[=+\-*/^<>≤≥≈]")
    for m in _INLINE_MATH_RE.finditer(text):
        span = m.group(1).strip()
        if not span:
            broken += 1
            continue
        # Bare subscript-only / superscript-only LaTeX: broken.
        if span.startswith(("\\_", "_{", "^{")) and not operator_re.search(span):
            broken += 1
            continue
        # Has an operator OR multiple distinct variable letters: real.
        var_chars = re.findall(r"[A-Za-z]", span)
        if operator_re.search(span) or len(set(var_chars)) >= 2:
            real += 1
        else:
            # Single variable or short fragment: treat as fragment.
            broken += 1
    return real, broken


def score(text: str, paper: str | None = None) -> dict[str, object]:
    real, broken = _classify_inline_math(text)
    out: dict[str, object] = {
        "inline_math": _count(_INLINE_MATH_RE, text),
        "inline_math_real": real,
        "inline_math_broken": broken,
        "block_math": _count(_BLOCK_MATH_RE, text),
        "math_symbols": _count(_MATH_SYMBOLS_RE, text),
    }
    if paper:
        if paper not in EQUATION_CHECKLISTS:
            raise ValueError(f"unknown paper: {paper}; choices: {list(EQUATION_CHECKLISTS)}")
        normalised = _normalise_for_equations(text)
        results: list[tuple[str, bool]] = []
        for label, pattern in EQUATION_CHECKLISTS[paper]:
            ok = bool(re.search(pattern, normalised, re.IGNORECASE))
            results.append((label, ok))
        out["equation_checklist"] = results
        out["equation_recall"] = (
            sum(1 for _, ok in results if ok),
            len(results),
        )
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Math recall metric for OCR outputs.")
    ap.add_argument(
        "--paper",
        choices=sorted(EQUATION_CHECKLISTS.keys()),
        default=None,
        help="Run a per-paper known-equations checklist in addition to counts.",
    )
    ap.add_argument("files", nargs="+", help="Markdown files to score.")
    args = ap.parse_args(argv)

    for path in args.files:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        result = score(text, paper=args.paper)
        print(f"== {path}")
        print(
            f"  inline_math:  {result['inline_math']} "
            f"(real: {result['inline_math_real']}, "
            f"broken: {result['inline_math_broken']})"
        )
        print(f"  block_math:   {result['block_math']}")
        print(f"  math_symbols: {result['math_symbols']}")
        if args.paper:
            for label, ok in result["equation_checklist"]:  # type: ignore[union-attr]
                marker = "[ok]  " if ok else "[miss]"
                print(f"  {marker} {label}")
            found, total = result["equation_recall"]  # type: ignore[misc]
            print(f"  equation_recall: {found}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
