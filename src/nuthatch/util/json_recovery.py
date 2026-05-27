# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Robust JSON recovery from LLM responses.

LLMs return JSON in many flavours: clean, fenced (```json ... ```),
prefixed with prose, trailing comma, single-quoted, truncated. This
module makes a best-effort to recover a `dict` from any of them.

Strategy (in order, returning the first successful parse):

  1. `json_repair.loads` if the library is installed. It handles most
     LLM JSON quirks (trailing commas, unquoted keys, partial output).
  2. Direct `json.loads` after stripping common code-fence wrappers.
  3. Pull the first `{...}` block via regex and parse it.

Returns `None` only when every strategy fails on completely
unrecoverable text. Eliminates 4 copies of `_try_parse_json` previously
duplicated across testset_generator / graph_eval / cluster_eval /
extract_concepts.
"""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_LEADING_FENCE_RE = re.compile(r"^```\w*\s*")
_TRAILING_FENCE_RE = re.compile(r"\s*```\s*$")


def parse_llm_json(text: str) -> dict[str, Any] | None:
    """Recover a `dict` from an LLM response. None if unrecoverable."""
    if not text:
        return None
    stripped = text.strip()

    # 1. json_repair handles most LLM JSON quirks if available.
    try:
        import json_repair

        repaired = json_repair.loads(stripped)
        if isinstance(repaired, dict):
            return repaired
    except ImportError:
        pass
    except Exception:
        # json_repair can throw on truly broken input; fall through.
        pass

    # 2. Strip code fences then try a direct parse.
    if stripped.startswith("```"):
        stripped = _LEADING_FENCE_RE.sub("", stripped)
        stripped = _TRAILING_FENCE_RE.sub("", stripped)
    try:
        loaded = json.loads(stripped)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    # 3. Greedy regex for the first {...} block.
    match = _JSON_BLOCK_RE.search(stripped)
    if match is None:
        return None
    try:
        loaded = json.loads(match.group(0))
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        return None
    return None
