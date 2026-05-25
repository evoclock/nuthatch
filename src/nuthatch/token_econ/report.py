# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Aggregation + markdown rendering of the token-economy log.

Purpose: turn the per-query records in `<corpus>/.kg/token_log.jsonl`
    into summary stats the user reads in Obsidian and the MCP
    `token_econ_report` tool returns as JSON.

Inputs at `aggregate`: an iterable of `TokenRecord`s (typically
    `TokenLog.iter_records(...)`) + a `group_by` axis.

Outputs: `ReportSummary` dataclass with totals + per-group rows.

`render_markdown_report` formats a `ReportSummary` as the markdown
that lands at `<corpus>/reports/token-economy-<date>.md` for
Obsidian Dataview / human reading. The format-template pattern is
reused from kestrel's `print_benchmark` (per-question table with a
reduction ratio column); nuthatch's version groups by axis and
totals at the bottom rather than running a fixed sample set.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from nuthatch.token_econ.log import TokenRecord

GroupBy = Literal["tool", "day", "surface"]


@dataclass(frozen=True)
class GroupRow:
    key: str
    n_queries: int
    tokens_served_total: int
    tokens_counterfactual_total: int
    reduction_ratio_avg: float


@dataclass(frozen=True)
class ReportSummary:
    n_queries: int
    tokens_served_total: int
    tokens_counterfactual_total: int
    reduction_ratio_overall: float
    tokens_saved_total: int
    pct_saved: float
    group_by: GroupBy
    groups: list[GroupRow] = field(default_factory=list)


def _group_key(record: TokenRecord, group_by: GroupBy) -> str:
    if group_by == "tool":
        return record.tool
    if group_by == "surface":
        return record.surface_id
    # day
    return (record.timestamp_utc or "")[:10] or "unknown"


def aggregate(
    records: Iterable[TokenRecord], *, group_by: GroupBy = "tool"
) -> ReportSummary:
    """Aggregate `records` into a `ReportSummary` grouped by `group_by`."""
    by_group: dict[str, list[TokenRecord]] = defaultdict(list)
    for r in records:
        by_group[_group_key(r, group_by)].append(r)

    groups: list[GroupRow] = []
    n_queries_total = 0
    tokens_served_total = 0
    tokens_counterfactual_total = 0
    ratio_weighted_sum = 0.0

    for key, group_records in sorted(by_group.items()):
        n = len(group_records)
        served = sum(r.tokens_served for r in group_records)
        counter = sum(r.tokens_counterfactual for r in group_records)
        if served > 0:
            ratio_avg = round(counter / served, 2)
        else:
            ratio_avg = 1.0
        groups.append(
            GroupRow(
                key=key,
                n_queries=n,
                tokens_served_total=served,
                tokens_counterfactual_total=counter,
                reduction_ratio_avg=ratio_avg,
            )
        )
        n_queries_total += n
        tokens_served_total += served
        tokens_counterfactual_total += counter
        ratio_weighted_sum += ratio_avg * n

    if tokens_served_total > 0:
        reduction_ratio_overall = round(
            tokens_counterfactual_total / tokens_served_total, 2
        )
        tokens_saved_total = tokens_counterfactual_total - tokens_served_total
        pct_saved = round(
            100.0 * tokens_saved_total / max(tokens_counterfactual_total, 1), 1
        )
    else:
        reduction_ratio_overall = 1.0
        tokens_saved_total = 0
        pct_saved = 0.0

    return ReportSummary(
        n_queries=n_queries_total,
        tokens_served_total=tokens_served_total,
        tokens_counterfactual_total=tokens_counterfactual_total,
        reduction_ratio_overall=reduction_ratio_overall,
        tokens_saved_total=tokens_saved_total,
        pct_saved=pct_saved,
        group_by=group_by,
        groups=groups,
    )


def summary_as_dict(summary: ReportSummary) -> dict[str, Any]:
    """JSON-shaped dict suitable for the MCP `token_econ_report` tool."""
    return {
        "n_queries": summary.n_queries,
        "tokens_served_total": summary.tokens_served_total,
        "tokens_counterfactual_total": summary.tokens_counterfactual_total,
        "reduction_ratio_overall": summary.reduction_ratio_overall,
        "tokens_saved_total": summary.tokens_saved_total,
        "pct_saved": summary.pct_saved,
        "group_by": summary.group_by,
        "groups": [
            {
                "key": g.key,
                "n_queries": g.n_queries,
                "tokens_served_total": g.tokens_served_total,
                "tokens_counterfactual_total": g.tokens_counterfactual_total,
                "reduction_ratio_avg": g.reduction_ratio_avg,
            }
            for g in summary.groups
        ],
    }


def render_markdown_report(
    summary: ReportSummary,
    *,
    corpus_name: str | None = None,
    title: str | None = None,
) -> str:
    """Render a `ReportSummary` as Obsidian-Dataview-friendly markdown.

    Format-template pattern reused from kestrel's `print_benchmark`:
    a small header with totals, a per-group table, a saved-tokens
    bottom line. Obsidian Dataview parses the YAML frontmatter for
    cross-corpus queries.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rendered_title = title or "Token economy report"
    front = [
        "---",
        f'title: "{rendered_title}"',
        f"generated: {now}",
    ]
    if corpus_name:
        front.append(f'corpus: "{corpus_name}"')
    front.extend(
        [
            f"group_by: {summary.group_by}",
            f"n_queries: {summary.n_queries}",
            f"tokens_served_total: {summary.tokens_served_total}",
            f"tokens_counterfactual_total: {summary.tokens_counterfactual_total}",
            f"reduction_ratio_overall: {summary.reduction_ratio_overall}",
            f"pct_saved: {summary.pct_saved}",
            "tags: [token_economy, report]",
            "---",
            "",
        ]
    )

    lines = list(front)
    lines.append(f"# {rendered_title}")
    lines.append("")
    lines.append(
        f"**{summary.n_queries} queries** served "
        f"**{summary.tokens_served_total:,} tokens** "
        f"vs **{summary.tokens_counterfactual_total:,} full-corpus tokens** "
        f"= **{summary.pct_saved}% saved** "
        f"({summary.reduction_ratio_overall}× reduction)."
    )
    lines.append("")
    lines.append(f"## Per-{summary.group_by} breakdown")
    lines.append("")
    if not summary.groups:
        lines.append("_no records in range_")
        lines.append("")
    else:
        lines.append(
            "| key | queries | served | counterfactual | reduction× |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for g in summary.groups:
            lines.append(
                f"| {g.key} | {g.n_queries} | {g.tokens_served_total:,} | "
                f"{g.tokens_counterfactual_total:,} | {g.reduction_ratio_avg} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
