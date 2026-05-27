# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Append-only JSONL log of per-query token-economy records.

Purpose: persistent record of every MCP query the surface served +
    its measured cost. Append-only so the file survives crashes;
    one record per line so streaming readers handle very-long logs
    without loading them whole.

Inputs at `append`: a `TokenRecord` (or compatible dict from
    `measure.measure_query`). At `iter_records`: a path filter
    (date range / tool / surface) optional.

Outputs: lines appended to `<corpus>/.kg/token_log.jsonl`.

Assumptions: file path is the corpus's `<root>/.kg/token_log.jsonl`
    by convention; the caller passes the explicit `Path` (no
    discovery in this module). One process writes at a time; on a
    multi-writer setup add OS-level locking at a higher layer.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenRecord:
    """Typed view of a single line in the token log."""

    timestamp_utc: str
    tool: str
    surface_id: str
    tokens_served: int
    tokens_counterfactual: int
    reduction_ratio: float
    tokenizer: str
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TokenRecord:
        known = {
            "timestamp_utc",
            "tool",
            "surface_id",
            "tokens_served",
            "tokens_counterfactual",
            "reduction_ratio",
            "tokenizer",
        }
        return cls(
            timestamp_utc=str(d["timestamp_utc"]),
            tool=str(d["tool"]),
            surface_id=str(d.get("surface_id", "unknown")),
            tokens_served=int(d["tokens_served"]),
            tokens_counterfactual=int(d["tokens_counterfactual"]),
            reduction_ratio=float(d["reduction_ratio"]),
            tokenizer=str(d.get("tokenizer", "unknown")),
            extras={k: v for k, v in d.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "timestamp_utc": self.timestamp_utc,
            "tool": self.tool,
            "surface_id": self.surface_id,
            "tokens_served": self.tokens_served,
            "tokens_counterfactual": self.tokens_counterfactual,
            "reduction_ratio": self.reduction_ratio,
            "tokenizer": self.tokenizer,
        }
        out.update(self.extras)
        return out


class TokenLog:
    """Append-only JSONL writer + reader for `<corpus>/.kg/token_log.jsonl`."""

    __slots__ = ("_path",)

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: TokenRecord | dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict() if isinstance(record, TokenRecord) else dict(record)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

    def iter_records(
        self,
        *,
        since: date | str | None = None,
        until: date | str | None = None,
        tool: str | None = None,
        surface_id: str | None = None,
    ) -> Iterator[TokenRecord]:
        since_d = _coerce_date(since)
        until_d = _coerce_date(until)
        if not self._path.is_file():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if tool and d.get("tool") != tool:
                    continue
                if surface_id and d.get("surface_id") != surface_id:
                    continue
                ts = _parse_timestamp_date(d.get("timestamp_utc", ""))
                if since_d and ts and ts < since_d:
                    continue
                if until_d and ts and ts > until_d:
                    continue
                yield TokenRecord.from_dict(d)

    def count(self) -> int:
        return sum(1 for _ in self.iter_records())


def _parse_timestamp_date(text: str) -> date | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _coerce_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    # Accept "YYYY-MM-DD" or full ISO-8601 ("YYYY-MM-DDTHH:MM:SSZ").
    cleaned = text.rstrip("Z").replace("Z", "")
    try:
        return datetime.fromisoformat(cleaned).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
