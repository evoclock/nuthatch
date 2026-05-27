# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Per-corpus configuration loader for `<corpus>/.kg/config.yaml`.

Purpose: let users override pipeline defaults (chunking, embedding,
    dedup thresholds, ...) on a per-corpus basis without editing
    source. The config file is optional; when absent, code defaults
    apply.

Convention: every top-level section is a module-scoped namespace
    (`embedding:`, `dedup:`, `clustering:`, ...). Code that reads
    config asks for a specific section and gets a typed dict back.

Tuning disclaimer (per DECISIONS.md): the shipped defaults are
    tuned for the release-shape corpus (academic papers / preprints
    / patents at the ~3000-char chunk size). Users who override
    these are responsible for tuning. nuthatch will accept their
    values without protest; whether the retrieval quality holds
    is on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding + chunking overrides. All fields optional."""

    max_chars: int | None = None
    overlap_chars: int | None = None
    batch_size: int | None = None
    model: str | None = None


@dataclass(frozen=True)
class SemanticExtractConfig:
    """Per-corpus overrides for the semantic-extract + augment-into-graph
    step. All fields optional; missing fields fall back to the defaults
    documented at each consumer (e.g. `add_semantic_edges` in
    `nuthatch.semantic_extract.augment`).

    semantic_edge_threshold: cosine similarity floor for adding a
        `shares_summary_with` edge between two documents whose summaries
        are similar. Lower -> more edges (denser graph, looser
        communities). Higher -> sparser graph, tighter communities.
        Default 0.55 in the augment library.
    """

    semantic_edge_threshold: float | None = None


@dataclass(frozen=True)
class CorpusConfig:
    """Parsed `<corpus>/.kg/config.yaml`. Sections are typed."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    semantic_extract: SemanticExtractConfig = field(
        default_factory=SemanticExtractConfig,
    )
    raw: dict[str, Any] = field(default_factory=dict)


def load_corpus_config(path: Path) -> CorpusConfig:
    """Load + parse `<corpus>/.kg/config.yaml`.

    Missing file -> empty `CorpusConfig` (all defaults apply).
    Malformed YAML -> empty `CorpusConfig` + log a warning at the
    caller's discretion (this loader does not raise).
    Unknown top-level keys are preserved in `raw` so future readers
    can pick them up without a schema migration.
    """
    if not path.is_file():
        return CorpusConfig()
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return CorpusConfig()
    if not isinstance(loaded, dict):
        return CorpusConfig()

    embed_raw = loaded.get("embedding") or {}
    if not isinstance(embed_raw, dict):
        embed_raw = {}
    embedding = EmbeddingConfig(
        max_chars=_int_or_none(embed_raw.get("max_chars")),
        overlap_chars=_int_or_none(embed_raw.get("overlap_chars")),
        batch_size=_int_or_none(embed_raw.get("batch_size")),
        model=_str_or_none(embed_raw.get("model")),
    )

    semex_raw = loaded.get("semantic_extract") or {}
    if not isinstance(semex_raw, dict):
        semex_raw = {}
    semantic_extract = SemanticExtractConfig(
        semantic_edge_threshold=_float_or_none(
            semex_raw.get("semantic_edge_threshold"),
        ),
    )

    return CorpusConfig(
        embedding=embedding,
        semantic_extract=semantic_extract,
        raw=loaded,
    )


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
