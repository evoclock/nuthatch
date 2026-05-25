# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Edge dataclasses + confidence labels for the knowledge graph.

Purpose: every edge nuthatch writes into the graph carries an
    explicit confidence label so downstream consumers (the LLM
    surface, the clustering layer, the analytics dashboard) can
    weight or filter by reliability without re-deriving provenance.

Inputs: at construction time, source + target node IDs plus a
    relation kind and a confidence tier.

Outputs: a frozen `Edge` dataclass instance with the metadata the
    `graph/build.py` consumer attaches as edge attributes in the
    NetworkX graph.

Pattern reused from kestrel's edge-schema convention (every edge
typed + confidence-labelled). nuthatch's confidence vocabulary is
deliberately three-tier:

- `EXTRACTED`: directly lifted from source text (e.g. a citation
  found in the references list, an author named in the metadata
  block). Highest confidence; structural ground truth.
- `INFERRED`: derived from a heuristic or model (NER entity
  co-mention, topic clustering, alias resolution). Medium
  confidence; the heuristic is documented and the score recorded.
- `AMBIGUOUS`: a match the extractor could not disambiguate (two
  authors with the same surname, two papers with the same title +
  different DOI). Low confidence; surfaced to the user when they
  query, never silently merged.

Assumptions: relations are typed strings (free-form for v1, may
    move to an enum once the relation vocabulary stabilises after
    Sprint 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Confidence(StrEnum):
    EXTRACTED = "EXTRACTED"
    INFERRED = "INFERRED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class Edge:
    """A single edge ready for `G.add_edge(source, target, **attrs)`."""

    source: str
    target: str
    relation: str
    confidence: Confidence = Confidence.EXTRACTED
    score: float | None = None  # populated for INFERRED edges
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_attrs(self) -> dict[str, Any]:
        """Return the dict of attributes networkx will store on the edge."""
        attrs: dict[str, Any] = {
            "relation": self.relation,
            "confidence": str(self.confidence),
        }
        if self.score is not None:
            attrs["score"] = float(self.score)
        if self.provenance:
            attrs["provenance"] = dict(self.provenance)
        return attrs
