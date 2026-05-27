# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities used across nuthatch subpackages and CLI subcommands.

These are intentionally small + general-purpose. Per `code-style.md`,
pipeline phases (ingest, embed, graph, clustering, concepts, eval, viz)
import from `nuthatch.util` instead of duplicating helpers. The CLI in
`src/nuthatch/cli.py` is the only outward face; subpackages stay pure.
"""

from nuthatch.util.corpus_resolve import resolve_corpus
from nuthatch.util.json_recovery import parse_llm_json
from nuthatch.util.palette import RELATION_COLORS, TYPE_COLORS
from nuthatch.util.timestamps import utc_tag

__all__ = [
    "RELATION_COLORS",
    "TYPE_COLORS",
    "parse_llm_json",
    "resolve_corpus",
    "utc_tag",
]
