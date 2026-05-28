# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: LicenseRef-MIT-Commercial-Attribution

"""Pre/post pipeline hooks for user-extensibility.

Purpose: let users plug functions into the ingest pipeline at named
    stages without modifying nuthatch itself. Hooks can rewrite
    extracted markdown (e.g. domain-specific OCR fixups), enrich
    metadata (look up a DOI's authors via Crossref), or perform
    side effects (notify a Slack channel, write a per-ingest log).

Inputs: a `HookRegistry` populated with `HookSpec`s the user
    registers. Each hook is identified by `(stage, callable)`.

Outputs: the modified context (markdown or metadata) returned by
    the chain of hooks for a given stage.

Pattern reused from `a prior implementation`'s `hooks.py` (the pre/post hook
registry in the external pipeline library nuthatch borrows from).
nuthatch's implementation is a fresh write: typed `HookStage` enum
matching the orchestrator's pipeline stages, an explicit registry
class users can populate, and contracts that pass + return typed
context objects (not opaque mutables).

Assumptions: hooks run in registration order and may raise to
    fail-fast. The orchestrator wraps hook calls in try/except so a
    crashing hook can be logged and routed to a quarantine path
    rather than killing the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class HookStage(StrEnum):
    """Pipeline stages a user hook can attach to."""

    POST_EXTRACT = "post_extract"
    POST_METADATA = "post_metadata"
    PRE_QUARANTINE = "pre_quarantine"
    POST_INGEST = "post_ingest"


# Each stage receives + returns a context dict the hook may mutate
# or replace. Hooks return the (possibly new) context.
HookFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class HookSpec:
    """Registration record for a single hook."""

    stage: HookStage
    fn: HookFn
    name: str = ""
    priority: int = 100  # lower runs earlier


@dataclass
class HookRegistry:
    """Registered hooks, keyed by stage."""

    _hooks: dict[HookStage, list[HookSpec]] = field(default_factory=dict)

    def register(self, spec: HookSpec) -> None:
        self._hooks.setdefault(spec.stage, []).append(spec)
        self._hooks[spec.stage].sort(key=lambda s: s.priority)

    def hooks(self, stage: HookStage) -> list[HookSpec]:
        return list(self._hooks.get(stage, []))

    def run(self, stage: HookStage, context: dict[str, Any]) -> dict[str, Any]:
        """Run every hook registered for `stage` in priority order.

        Each hook receives the current context and returns the
        possibly-updated context. A hook may raise; the caller is
        responsible for catching, logging, and routing to a
        quarantine path if needed.
        """
        for spec in self.hooks(stage):
            context = spec.fn(context)
        return context

    def clear(self, stage: HookStage | None = None) -> None:
        if stage is None:
            self._hooks.clear()
        else:
            self._hooks.pop(stage, None)
