# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Decay + supersession pass (Sprint 8).

Ties the graph-layer decay kernel in `nuthatch.graph.decay` together
with on-disk card frontmatter and the supersession scanner. The
pass runs periodically (user-triggered via `nuthatch decay`) and is
idempotent: running it twice in a row over an unchanged corpus
produces the same outputs.
"""

from nuthatch.decay.frontmatter import (
    parse_card,
    update_frontmatter,
    write_card,
)
from nuthatch.decay.pass_ import (
    ArchiveCandidate,
    DecayResult,
    SupersessionEvent,
    render_report,
    run_decay_pass,
)
from nuthatch.decay.supersession import (
    SupersessionPair,
    apply_supersession,
    find_supersession_pairs,
)

__all__ = [
    "ArchiveCandidate",
    "DecayResult",
    "SupersessionEvent",
    "SupersessionPair",
    "apply_supersession",
    "find_supersession_pairs",
    "parse_card",
    "render_report",
    "run_decay_pass",
    "update_frontmatter",
    "write_card",
]
