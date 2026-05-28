# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Per-query token measurement for the token-economy log.

Purpose: take what a single MCP tool call SERVED to the consuming
    surface and compute how many tokens it represents, alongside the
    counterfactual cost of having sent the WHOLE corpus instead. The
    ratio is nuthatch's user-facing differentiator: "you saved X
    tokens this query, Y% reduction vs full-corpus naive ingest".

Inputs at `measure_query`: the tool name, the served text (what the
    MCP tool returned), the corpus-size source (either a raw text
    blob for the counterfactual, or a precomputed token count), and
    operational metadata (surface_id, timestamp).

Outputs: a `TokenRecord`-compatible dict that the caller appends to
    the corpus's `TokenLog`.

Pattern adapted from a prior knowledge-graph implementation; original lived in `benchmark.py` (the `_estimate_tokens`
shape and the reduction-ratio framing). nuthatch replaces a prior implementation's
character-count approximation (`chars / 4`) with `tiktoken`'s
`cl100k_base` encoding for ~10-30% better accuracy. Tiktoken is the
universal approximation here; per-model precision is a later
extension.

Assumptions: tiktoken is installed (it is, since Sprint 3). On hosts
    where it isn't available, `count_tokens` falls back to the
    `chars / 4` rule and records the fallback in the record's
    `tokenizer` field so downstream consumers can weight accordingly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# tiktoken's encoding for modern OpenAI / Anthropic-style models. The
# vocabularies differ but `cl100k_base` is the closest universal
# approximation. The `tokenizer` field on every record carries the
# encoding name so a future per-model precision pass can reconcile.
DEFAULT_ENCODING: str = "cl100k_base"

# Fallback constant for the no-tiktoken path. Same value a prior implementation uses.
_CHARS_PER_TOKEN_FALLBACK: int = 4


def _import_tiktoken() -> Any:
    try:
        import tiktoken

        return tiktoken
    except ImportError:
        return None


def count_tokens(
    text: str,
    *,
    encoding_name: str = DEFAULT_ENCODING,
) -> tuple[int, str]:
    """Return `(token_count, tokenizer_label)` for `text`.

    `tokenizer_label` is the encoding name on success or
    `'chars_per_4'` when tiktoken is unavailable. The caller
    persists the label so downstream aggregation can flag
    approximate counts.
    """
    if not text:
        return 0, encoding_name
    tiktoken = _import_tiktoken()
    if tiktoken is None:
        return max(1, len(text) // _CHARS_PER_TOKEN_FALLBACK), "chars_per_4"
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except (ValueError, KeyError):
        return max(1, len(text) // _CHARS_PER_TOKEN_FALLBACK), "chars_per_4"
    return len(enc.encode(text)), encoding_name


def measure_query(
    *,
    tool: str,
    served_text: str,
    counterfactual_text: str | None = None,
    counterfactual_tokens: int | None = None,
    surface_id: str = "unknown",
    encoding_name: str = DEFAULT_ENCODING,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Compute the per-query token record.

    Provide one of `counterfactual_text` (the full text the agent
    would have ingested without nuthatch — typically the corpus
    body concat) or `counterfactual_tokens` (a precomputed count,
    cheaper when the corpus is large and re-tokenising on every
    query is wasteful). When neither is supplied the counterfactual
    is recorded as 0 and `reduction_ratio` is 1.0 (no claim).
    """
    served_tokens, tokenizer = count_tokens(served_text, encoding_name=encoding_name)
    if counterfactual_tokens is None and counterfactual_text is not None:
        counterfactual_tokens, _ = count_tokens(counterfactual_text, encoding_name=encoding_name)
    if counterfactual_tokens is None:
        counterfactual_tokens = 0

    reduction_ratio = round(counterfactual_tokens / served_tokens, 2) if served_tokens > 0 else 1.0

    ts = timestamp or datetime.now(UTC)
    return {
        "timestamp_utc": ts.isoformat(),
        "tool": tool,
        "surface_id": surface_id,
        "tokens_served": served_tokens,
        "tokens_counterfactual": counterfactual_tokens,
        "reduction_ratio": reduction_ratio,
        "tokenizer": tokenizer,
    }
