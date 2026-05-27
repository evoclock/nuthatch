# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""LLM-naming pass over a corpus's persisted community indices.

Purpose: replace the heuristic word-frequency labels in
    `.kg/communities.json` (canonical) and any `.kg/communities_<suffix>.json`
    siblings with topical 2-4 word names produced by an LLM, given each
    community's member titles + tags. Heuristic labels look like
    "circadian gene expression mouse"; LLM labels look like
    "Circadian Molecular Biology".

Inputs:
    - paths: list of `.kg/communities*.json` files to rewrite in place
    - cards_dir: where to find `<doc_id>.md` cards for title + tag lookup
    - llm_invoke: a callable `(prompt: str) -> str` returning raw model
        output. Pass `None` to use the default ollama-backed builder
        configured via `nuthatch.util.llm_backends.build_llm`.

Outputs:
    Rewrites each input file in place, replacing only the `labels` field.
    All other fields (flat, hierarchy, members, core_nodes, centroids
    metadata) are preserved byte-for-byte besides the relabel diff.

Assumptions:
    - Cards under `cards_dir` have YAML frontmatter with `title:` and
      flat `tags:` list. Missing cards degrade label quality but never
      raise.
    - Member IDs in the persisted indices may carry the graph node
      prefix `doc::`; we strip before card lookup.

Failure modes:
    - LLM call fails for one community: keep the existing heuristic
      label for that community and continue.
    - Cards directory missing: fall back to using member ids as titles;
      label quality degrades but the pass completes.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Default model used when the CLI flag `--relabel-llm` is passed
# without an explicit `--relabel-model`. Cheap + local; runs on
# consumer hardware via Ollama.
DEFAULT_RELABEL_MODEL: str = "granite3-dense:8b"
DEFAULT_RELABEL_BACKEND: str = "ollama"

# Generation budget for the relabel call. 64 tokens is generous for a
# 2-4 word title; we cap at 6 words in post-processing anyway.
_RELABEL_NUM_PREDICT: int = 64

# Low temperature keeps labels deterministic across runs; the goal is
# stable naming, not creativity.
_RELABEL_TEMPERATURE: float = 0.1

# How many member titles to include in the prompt context. 12 is enough
# to surface the dominant theme without ballooning prompt tokens.
_MAX_MEMBERS_IN_PROMPT: int = 12

_PROMPT = """You are naming a cluster of academic papers. Given the titles and tags below, output a 2-4 word topical name that captures what the cluster is about.

Rules:
- 2-4 words, Title Case
- No quotes, no punctuation, no preamble
- Output the name only, nothing else
- Prefer the dominant theme over a literal title quote

Papers in this cluster:
{members}

Topical name:"""


def _read_card_meta(card_path: Path) -> dict[str, Any]:
    """Pull title + tags from a card's YAML frontmatter.

    Hand-parsed: pyyaml is not a hard dep of this module and the
    frontmatter shape is fixed (`---` ... `---` at file top, simple
    `key: value` or `key:` followed by `- item` lines). Returns
    `{"title": "", "tags": []}` on any parse failure.
    """
    out: dict[str, Any] = {"title": "", "tags": []}
    if not card_path.is_file():
        return out
    text = card_path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return out
    end = text.find("\n---", 4)
    if end < 0:
        return out
    fm = text[4:end]
    cur_key: str | None = None
    for raw in fm.splitlines():
        line = raw.rstrip()
        if not line:
            cur_key = None
            continue
        if line.startswith("- ") and cur_key == "tags":
            out["tags"].append(line[2:].strip())
        elif ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            cur_key = k.strip()
            val = v.strip()
            if cur_key == "title" and val:
                out["title"] = val
            elif cur_key == "tags" and val and val.startswith("[") and val.endswith("]"):
                # inline list form (rare in our cards): tags: [a, b, c]
                out["tags"] = [t.strip() for t in val[1:-1].split(",") if t.strip()]
    return out


def _build_member_block(
    member_ids: list[str],
    cards_dir: Path,
    *,
    max_members: int = _MAX_MEMBERS_IN_PROMPT,
) -> str:
    """Render up to `max_members` lines of `- TITLE (tag1, tag2, ...)`."""
    lines: list[str] = []
    for raw_id in member_ids[:max_members]:
        # Members in communities_*.json carry the graph node prefix
        # `doc::`; card filenames omit it. Strip before lookup.
        doc_id = raw_id[len("doc::") :] if raw_id.startswith("doc::") else raw_id
        meta = _read_card_meta(cards_dir / f"{doc_id}.md")
        title = str(meta["title"]).strip() or doc_id
        tags = ", ".join(meta["tags"][:5])
        if tags:
            lines.append(f"- {title} ({tags})")
        else:
            lines.append(f"- {title}")
    if len(member_ids) > max_members:
        lines.append(f"(and {len(member_ids) - max_members} more)")
    return "\n".join(lines)


def _clean_label(raw: str) -> str:
    """Strip preamble / quotes / punctuation from the LLM's response.

    Caps at 6 words to defend against runaway outputs; the prompt asks
    for 2-4 but cheap local models occasionally emit short paragraphs.
    """
    s = raw.strip()
    for prefix in ("Topical name:", "Name:", "Label:", "Cluster:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix) :].strip()
    s = s.strip("\"'`")
    s = s.splitlines()[0].strip() if s else s
    words = s.split()
    if len(words) > 6:
        s = " ".join(words[:6])
    return s


def _label_one(
    llm_invoke: Callable[[str], str],
    community_id: int,
    member_ids: list[str],
    cards_dir: Path,
    old_label: str,
) -> str:
    """Produce a label for one community; fall back to `old_label` on error."""
    if not member_ids:
        return old_label
    block = _build_member_block(member_ids, cards_dir)
    prompt = _PROMPT.format(members=block)
    try:
        raw = llm_invoke(prompt)
        new = _clean_label(raw)
        if not new:
            return old_label
        return new
    except Exception as exc:
        print(
            f"[relabel] community {community_id}: LLM failed ({exc!s}); keeping heuristic label.",
            file=sys.stderr,
        )
        return old_label


def relabel_index_file(
    path: Path,
    cards_dir: Path,
    llm_invoke: Callable[[str], str],
) -> dict[int, tuple[str, str]]:
    """Rewrite one `communities*.json` in place with LLM labels.

    Returns a mapping `{community_id: (old_label, new_label)}` for
    every community processed. Empty dict when the file does not exist.
    All non-label fields are preserved.
    """
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    members: dict[str, list[str]] = payload.get("members") or {}
    old_labels: dict[str, str] = payload.get("labels") or {}

    diff: dict[int, tuple[str, str]] = {}
    new_labels: dict[str, str] = {}
    for cid_str, member_ids in members.items():
        old = str(old_labels.get(cid_str, ""))
        new = _label_one(
            llm_invoke,
            int(cid_str),
            list(member_ids),
            cards_dir,
            old,
        )
        new_labels[cid_str] = new
        diff[int(cid_str)] = (old, new)

    payload["labels"] = new_labels
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    return diff


def relabel_communities(
    paths: list[Path],
    cards_dir: Path,
    *,
    llm_invoke: Callable[[str], str],
) -> dict[Path, dict[int, tuple[str, str]]]:
    """Apply the LLM relabel pass to every file in `paths`.

    Pure function: caller supplies the LLM-invocation callable so this
    can be tested with a stub. Returns per-file diff dicts so the CLI
    can pretty-print and tests can assert.
    """
    result: dict[Path, dict[int, tuple[str, str]]] = {}
    for p in paths:
        result[p] = relabel_index_file(p, cards_dir, llm_invoke)
    return result


def discover_index_paths(kg_dir: Path) -> list[Path]:
    """Find every `communities*.json` under `<corpus>/.kg/`.

    Always lists the canonical name first (if present) so per-file
    iteration order is stable: canonical, then suffixed siblings in
    lexicographic order. Returns an empty list if no files exist.
    """
    canonical = kg_dir / "communities.json"
    suffixed = sorted(p for p in kg_dir.glob("communities_*.json") if p.is_file())
    paths: list[Path] = []
    if canonical.is_file():
        paths.append(canonical)
    paths.extend(suffixed)
    return paths


def build_default_invoke(
    *,
    backend: str = DEFAULT_RELABEL_BACKEND,
    model: str = DEFAULT_RELABEL_MODEL,
    temperature: float = _RELABEL_TEMPERATURE,
    num_predict: int = _RELABEL_NUM_PREDICT,
) -> Callable[[str], str]:
    """Wrap `nuthatch.util.llm_backends.build_llm` into a plain callable.

    Returned callable takes a prompt string and returns the raw model
    text (post-extracted from langchain message objects). Module
    imports kept inside the function so tests that supply their own
    `llm_invoke` do not need langchain installed.
    """
    from nuthatch.util.llm_backends import build_llm

    llm = build_llm(
        backend=backend,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
    )

    def _invoke(prompt: str) -> str:
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

    return _invoke
