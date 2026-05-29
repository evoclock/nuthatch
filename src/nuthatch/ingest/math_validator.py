# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Inline-math validator for extracted markdown.

Classifies `$...$` spans in a Docling-extracted markdown body
as either real equations or broken fragments (subscript /
superscript with the host variable stripped, the classic
Docling-VLM failure mode on math-heavy papers).

Used by `IngestOrchestrator` when `--skip-chandra` is set: every
ingested doc gets validated, and docs with too many broken spans
are appended to `<corpus>/.kg/math_retry.jsonl` for a later
batch-Chandra pass to patch (see `docs/DECISIONS.md` and the
post-Thursday strand notes).

The classifier itself is the same shape as
`scripts/bench/math_recall.py:_classify_inline_math`, lifted into
the package so it can be imported as a library function rather
than re-implemented per caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^$\n]{1,200}?)\$(?!\$)")
_OPERATOR_RE = re.compile(r"[=+\-*/^<>≤≥≈]")

# A span that is JUST a bare subscript / superscript with nothing
# else is broken. Matches `\_{x}`, `_{ab}`, `^{6}`, `^{xyz}`. The
# operator check below otherwise misclassifies `^{6}` as real
# because `^` is in `_OPERATOR_RE`.
_BARE_SUB_SUP_RE = re.compile(r"^(?:\\_|_|\^)\{[^}]+\}$")

# Threshold defaults for "this doc needs Chandra retry".
# Tuned so a paper with 1-2 broken fragments doesn't trigger
# (some Docling fallibility is expected even on prose-only papers).
# A real math-heavy paper that Docling mangled will produce
# tens of broken spans.
_DEFAULT_BROKEN_THRESHOLD: int = 5
_DEFAULT_BROKEN_RATIO_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class MathValidationResult:
    """Outcome of `validate_math()` on a markdown body."""

    real_count: int
    broken_count: int
    broken_spans: list[str]
    needs_retry: bool
    threshold_broken: int = _DEFAULT_BROKEN_THRESHOLD
    threshold_ratio: float = _DEFAULT_BROKEN_RATIO_THRESHOLD

    @property
    def total_count(self) -> int:
        return self.real_count + self.broken_count

    @property
    def broken_ratio(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.broken_count / self.total_count


def classify_inline_math(text: str) -> tuple[int, int, list[str]]:
    """Return `(real_equations, broken_fragments, broken_span_samples)`.

    A real equation contains either an operator (=, +, -, *, /, ^,
    <, >) or two distinct variable letters. A broken fragment is a
    bare subscript / superscript span (`\\_{s}`, `^{6}`, `_{ab}`)
    or a single-character span with no operator.

    Spans are capped at 200 chars to avoid quadratic backtracking
    on pathological inputs.
    """
    real = 0
    broken = 0
    broken_samples: list[str] = []
    for m in _INLINE_MATH_RE.finditer(text):
        span = m.group(1).strip()
        if not span:
            broken += 1
            broken_samples.append(span)
            continue
        # Bare `^{...}` / `_{...}` / `\_{...}` is always broken even
        # though `^` is in `_OPERATOR_RE`. Must be checked before the
        # operator-presence heuristic.
        if _BARE_SUB_SUP_RE.match(span):
            broken += 1
            broken_samples.append(span)
            continue
        var_chars = re.findall(r"[A-Za-z]", span)
        if _OPERATOR_RE.search(span) or len(set(var_chars)) >= 2:
            real += 1
        else:
            broken += 1
            broken_samples.append(span)
    return real, broken, broken_samples


def validate_math(
    markdown: str,
    *,
    broken_threshold: int = _DEFAULT_BROKEN_THRESHOLD,
    broken_ratio_threshold: float = _DEFAULT_BROKEN_RATIO_THRESHOLD,
    sample_limit: int = 50,
) -> MathValidationResult:
    """Decide whether `markdown` needs a Chandra retry for its math.

    A doc needs retry when BOTH:

    - broken_count >= `broken_threshold`  (absolute count gate)
    - broken_count / (broken_count + real_count) >=
      `broken_ratio_threshold` (don't flag papers that just have a
      few broken spans amidst many real ones — that's noise, not a
      systematic failure)
    """
    real, broken, samples = classify_inline_math(markdown)
    total = real + broken
    ratio = 0.0 if total == 0 else broken / total
    needs_retry = broken >= broken_threshold and ratio >= broken_ratio_threshold
    return MathValidationResult(
        real_count=real,
        broken_count=broken,
        broken_spans=samples[:sample_limit],
        needs_retry=needs_retry,
        threshold_broken=broken_threshold,
        threshold_ratio=broken_ratio_threshold,
    )
