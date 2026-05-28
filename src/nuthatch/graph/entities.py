# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Entity extraction from extracted-paper text.

Purpose: lift the entities that become graph nodes out of a paper's
    body and metadata. Three sources, each producing typed entities
    with provenance:

    - **Authors** (from extracted metadata): one entity per author.
    - **Citations** (from Docling's reference list parser, or
      bracketed-citation regex on the body): one entity per cited
      work, plus the citation edge from paper -> cited work.
    - **Named entities** (NER via an optional spaCy or sentence-
      transformers pass): persons, organisations, methods, gene
      names, places. Optional — the corpus config opts in.

Inputs: extracted markdown + metadata dict.

Outputs: a list of `ExtractedEntity` with type, name, normalised
    key (id-safe slug), and provenance.

The bracketed-citation regex pattern is reused from PhD KB's
`retrofit_wiki_graph.py` (`[Author YEAR]` → `(surname, year)` →
stem lookup). nuthatch's implementation extends it: also recognise
DOI-style citations and footnote-style superscripts, and produce
entities that the edge layer can connect to the canonical paper
node when a match exists in the corpus.

NER is pluggable. The default extractor does NOT call spaCy or
sentence-transformers — those are optional dependencies. A corpus
config setting `entities.ner_backend = "spacy"` opts in.

Assumptions: heuristic-first, opt-in for heavier extractors.
    Confidence labels track the source: metadata-derived authors
    are `EXTRACTED`; NER mentions are `INFERRED` with a score.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from nuthatch.graph.edges import Confidence


class EntityType(StrEnum):
    AUTHOR = "author"
    CITATION = "citation"
    TOPIC = "topic"
    PERSON = "person"
    ORG = "org"
    METHOD = "method"
    GENE = "gene"
    PLACE = "place"
    OTHER = "other"


@dataclass(frozen=True)
class ExtractedEntity:
    type: EntityType
    name: str
    key: str  # id-safe slug for graph node ID
    confidence: Confidence = Confidence.EXTRACTED
    score: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_AUTHOR_YEAR_CITATION_RE = re.compile(r"\[([A-Z][A-Za-z_'-]+?)\s+(\d{4})[a-z]?\]")
_DOI_CITATION_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)


def _slug(text: str) -> str:
    s = _SLUG_RE.sub("_", (text or "").lower().strip()).strip("_")
    return s[:100] or "unknown"


def _author_key(name: str) -> str:
    return f"author::{_slug(name)}"


def _citation_key_author_year(surname: str, year: str) -> str:
    return f"citation::{_slug(surname)}_{year}"


def _citation_key_doi(doi: str) -> str:
    return f"citation::doi_{_slug(doi)}"


def _topic_key(topic: str) -> str:
    return f"topic::{_slug(topic)}"


class EntityExtractor:
    """Default heuristic entity extractor.

    Subclass and override `extract_ner` to plug in a spaCy or
    sentence-transformers NER pass. The default does no NER; only
    structural extraction (authors, citations, topics).
    """

    def extract(
        self,
        *,
        markdown: str,
        metadata: dict[str, Any],
    ) -> list[ExtractedEntity]:
        """Return all entities the heuristic pass can identify."""
        out: list[ExtractedEntity] = []
        out.extend(self._authors_from_metadata(metadata))
        out.extend(self._topics_from_metadata(metadata))
        out.extend(self._citations_from_body(markdown))
        out.extend(self.extract_ner(markdown))
        return out

    def extract_ner(self, _markdown: str) -> list[ExtractedEntity]:
        """Default no-op NER. Override in a subclass to plug in spaCy etc."""
        return []

    @staticmethod
    def _authors_from_metadata(meta: dict[str, Any]) -> list[ExtractedEntity]:
        raw = meta.get("authors") or []
        if isinstance(raw, str):
            raw = [raw]
        out: list[ExtractedEntity] = []
        for author in raw:
            name = str(author).strip()
            if not name:
                continue
            out.append(
                ExtractedEntity(
                    type=EntityType.AUTHOR,
                    name=name,
                    key=_author_key(name),
                    confidence=Confidence.EXTRACTED,
                    provenance={"source": "metadata.authors"},
                )
            )
        return out

    @staticmethod
    def _topics_from_metadata(meta: dict[str, Any]) -> list[ExtractedEntity]:
        raw = meta.get("topics") or []
        if isinstance(raw, str):
            raw = [t.strip() for t in raw.split(",")]
        out: list[ExtractedEntity] = []
        for topic in raw:
            name = str(topic).strip()
            if not name:
                continue
            out.append(
                ExtractedEntity(
                    type=EntityType.TOPIC,
                    name=name,
                    key=_topic_key(name),
                    confidence=Confidence.EXTRACTED,
                    provenance={"source": "metadata.topics"},
                )
            )
        return out

    @staticmethod
    def _citations_from_body(markdown: str) -> list[ExtractedEntity]:
        seen: set[str] = set()
        out: list[ExtractedEntity] = []
        # Bracketed `[Author YEAR]` citations (PhD KB-style).
        for m in _AUTHOR_YEAR_CITATION_RE.finditer(markdown):
            surname, year = m.group(1), m.group(2)
            key = _citation_key_author_year(surname, year)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ExtractedEntity(
                    type=EntityType.CITATION,
                    name=f"{surname} {year}",
                    key=key,
                    confidence=Confidence.EXTRACTED,
                    provenance={"source": "body.bracketed_citation"},
                )
            )
        # DOI references in the body.
        for m in _DOI_CITATION_RE.finditer(markdown):
            doi = m.group(1)
            key = _citation_key_doi(doi)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ExtractedEntity(
                    type=EntityType.CITATION,
                    name=doi,
                    key=key,
                    confidence=Confidence.EXTRACTED,
                    provenance={"source": "body.doi", "doi": doi},
                )
            )
        return out


def deduplicate_entities(entities: Iterable[ExtractedEntity]) -> list[ExtractedEntity]:
    """Collapse entities with the same `key`; keep the highest-confidence record.

    Used at graph-build time so two papers mentioning the same
    author / citation contribute a single graph node.
    """
    by_key: dict[str, ExtractedEntity] = {}
    confidence_rank = {
        Confidence.EXTRACTED: 0,
        Confidence.INFERRED: 1,
        Confidence.AMBIGUOUS: 2,
    }
    for e in entities:
        prior = by_key.get(e.key)
        if prior is None:
            by_key[e.key] = e
            continue
        if confidence_rank[e.confidence] < confidence_rank[prior.confidence]:
            by_key[e.key] = e
    return list(by_key.values())
