#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Script: key_facts

Path: scripts/bench/key_facts.py

Purpose: per-paper checklist scorer for OCR extraction benchmarks.
    Given a paper id and an OCR markdown output, check whether each
    known key fact appears in the output. Binary score per fact; the
    aggregate is `n_found / n_total`.

Inputs:
    --paper {mk1991, mendel}: which paper's checklist to use.
    <file.md>: OCR markdown output path.

Outputs: stdout, one line per fact: `[ok|miss]\\t<fact>`, then a
    trailing summary line: `score: N/M  (X.X%)`.

Assumptions: the OCR output is a single markdown string. Fact-match
    rules are case-insensitive substring or regex matches per fact.

Parameters: the per-paper FACTS dicts. Add a new paper by appending
    its entry to FACTS.

Failure Modes: a fact whose ground-truth string is mis-typed will
    show as `miss` for every tool; verify against the source PDF
    before adjusting.

Author: Julen Gamboa

Created: 2026-05-25

Last Edited: 2026-05-25 by Julen Gamboa
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Strip formatting that varies across tools so matchers compare content,
# not surface syntax. HTML tags and markdown emphasis marks both
# interrupt substring matching even when the underlying text is correct.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_EMPHASIS_CHARS = str.maketrans({"*": " ", "_": " ", "`": " ", "~": " "})
_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Strip HTML tags + markdown emphasis, collapse whitespace."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.translate(_MD_EMPHASIS_CHARS)
    return _WHITESPACE_RE.sub(" ", text)


# Per-paper key facts. Each entry is (label, matcher_callable).
# `matcher_callable(normalised_text: str) -> bool`.
def _contains(needle: str):
    n = needle.lower()
    return lambda t: n in t.lower()


def _matches(pattern: str):
    rx = re.compile(pattern, re.IGNORECASE)
    return lambda t: bool(rx.search(t))


FACTS = {
    "mk1991": [
        (
            "title: Adaptive protein evolution at the Adh locus in Drosophila",
            _contains("Adaptive protein evolution at the Adh locus"),
        ),
        ("author: McDonald", _contains("McDonald")),
        ("author: Kreitman (allow 1-letter OCR drift)", _matches(r"Krei?(t|m)?mit?man|Kreit?man")),
        ("institution: Princeton", _contains("Princeton")),
        ("Replacement Fixed = 7", _matches(r"Replacement.{0,40}\b7\b|\b7\b.{0,40}Replacement")),
        ("Synonymous Fixed = 17", _matches(r"Synonymous.{0,40}\b17\b|\b17\b.{0,40}Synonymous")),
        ("G-test stat: G = 7.43", _matches(r"G\s*=\s*7\.43")),
        ("p-value: P = 0.006", _matches(r"P\s*=\s*0\.00[56]")),
        ("body: 'amino-acid sequence'", _contains("amino-acid")),
        ("body: 'neutral mutations'", _contains("neutral mutation")),
    ],
    "wright1931": [
        (
            "title: Evolution in Mendelian Populations",
            _contains("Evolution in Mendelian Populations"),
        ),
        ("author: Sewall Wright", _matches(r"Sewall\s+Wright")),
        ("institution: University of Chicago", _contains("University of Chicago")),
        ("journal: Genetics", _contains("Genetics")),
        ("year: 1931", _matches(r"\b1931\b")),
        ("concept: random drift / genetic drift", _matches(r"random\s+drift|genetic\s+drift")),
        (
            "concept: effective N (Wright's notation)",
            _matches(
                r"effective\s+\$?N\$?\b|effective\s+number|effective\s+population|effective\s+size"
            ),
        ),
        ("concept: fixation", _contains("fixation")),
        (
            "concept: selection coefficient",
            _matches(r"selection\s+coefficient|coefficient\s+of\s+selection|\bs\s*="),
        ),
        ("body: 'mutation pressure'", _matches(r"mutation\s+pressure")),
        ("body: 'gene frequency' / 'gene frequencies'", _matches(r"gene\s+frequenc(y|ies)")),
    ],
    "mendel": [
        ("title: Plant Hybridi[sz]ation", _matches(r"plant[- ]hybridi[sz]ation")),
        ("author: Mendel", _contains("Mendel")),
        ("translator: Bateson (1909)", _contains("Bateson")),
        ("Brunn / Brno", _matches(r"Br(u|ü)nn|Brno")),
        ("seven characters / traits", _matches(r"seven\s+(characters|traits|differentiating)")),
        ("plant: Pisum (peas)", _matches(r"Pisum|pea[\s-]*plant|peas")),
        (
            "ratio: 3:1 (any rendering)",
            _matches(r"\b3\s*[:.]\s*1\b|\bthree\s+to\s+one\b|\b3\s+to\s+1\b"),
        ),
        ("dominant / recessive", _matches(r"dominant|recessive")),
        ("body: 'artificial fertilisation'", _matches(r"artificial\s+ferti(l|li)sat?ion")),
        (
            "body: 'F1' / 'first generation'",
            _matches(r"\bF1\b|first[\s-]+generation|first[\s-]+hybrid"),
        ),
    ],
}


def score(paper: str, text: str) -> tuple[int, int, list[tuple[str, bool]]]:
    if paper not in FACTS:
        raise ValueError(f"unknown paper: {paper}; choices: {list(FACTS)}")
    checklist = FACTS[paper]
    normalised = _normalise(text)
    results = [(label, matcher(normalised)) for label, matcher in checklist]
    found = sum(1 for _, ok in results if ok)
    return found, len(results), results


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Per-paper checklist scorer for OCR extraction benchmarks."
    )
    ap.add_argument("--paper", required=True, choices=list(FACTS))
    ap.add_argument("file", help="path to OCR markdown output")
    args = ap.parse_args(argv)

    text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    found, total, results = score(args.paper, text)
    for label, ok in results:
        marker = "[ok]  " if ok else "[miss]"
        print(f"{marker}\t{label}")
    pct = 100 * found / total
    print(f"score: {found}/{total}  ({pct:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
