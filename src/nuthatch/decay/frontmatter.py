# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""YAML-frontmatter read / write for per-doc cards.

Purpose: the decay pass needs to round-trip cards on disk
    (`<corpus>/cards/<doc_id>.md`) without disturbing their body
    content. Card writes happen in `nuthatch.render.card.render_card`
    at ingest time; this module handles the in-place updates the
    decay pass needs to apply (relevance, last_touched, status).

Assumptions: cards begin with a `---`-delimited YAML block as
    written by `render_card`. Cards missing frontmatter are
    returned as `({}, full_body_text)` and never written; the
    caller decides whether that's an error.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_FENCE = "---"


def parse_card(path: Path) -> tuple[dict[str, Any], str]:
    """Return `(frontmatter_dict, body_text)` for `path`.

    Returns `({}, full_text)` if no frontmatter fence is present;
    the caller decides whether that's a bug or a legitimate
    fronmatter-less file.
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_FENCE):
        return {}, text
    # Split into [empty, yaml_block, body]. A card produced by
    # render_card always ends the frontmatter with a "---\n" on its
    # own line, so the split-on-fence yields at least 3 parts.
    parts = text.split(f"\n{_FENCE}\n", 1)
    if len(parts) != 2:
        return {}, text
    yaml_block = parts[0].removeprefix(_FENCE).lstrip("\n")
    body = parts[1]
    loaded = yaml.safe_load(yaml_block) or {}
    if not isinstance(loaded, dict):
        return {}, text
    return loaded, body


def update_frontmatter(
    front: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a new dict with `updates` merged into `front`.

    `None` values in `updates` are written through, so callers can
    explicitly clear a field by setting it to `None`. (To remove a
    field entirely, drop it from the merged dict afterwards.)
    """
    merged = dict(front)
    merged.update(updates)
    return merged


def write_card(
    path: Path,
    frontmatter: Mapping[str, Any],
    body: str,
) -> None:
    """Write `path` with `frontmatter` as YAML block + `body`.

    Field ORDER is preserved from the frontmatter mapping. The body
    is written verbatim; if it had leading whitespace before this
    rewrite, it still does after.
    """
    yaml_text = yaml.dump(dict(frontmatter), default_flow_style=False, sort_keys=False).rstrip()
    path.write_text(
        f"{_FENCE}\n{yaml_text}\n{_FENCE}\n{body}",
        encoding="utf-8",
    )
