#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Standalone LLM relabel of an existing corpus's community indices.

Thin shim over `nuthatch.clustering.relabel`. The same logic is invoked
inline via `nuthatch cluster --relabel-llm`; this script is the
out-of-band path: re-relabel an existing corpus without re-clustering
(e.g. swapping models, fixing a bad relabel run, working against a KB
demo where the upstream cluster step is not available).

Inputs:
    --corpus <name|path>     corpus to operate on
    --index sbm|leiden|embeddings|all|canonical
                             which on-disk files to rewrite
    --backend ollama|openai|anthropic
                             LLM backend (default: ollama)
    --model <id>             model id (default: granite3-dense:8b)
    --temperature <float>    generation temperature (default: 0.1)

Outputs:
    Rewrites `<corpus>/.kg/communities*.json` files in place. Prints
    per-community old vs new label diff to stdout.

Re-run `nuthatch render --corpus <name>` after this to refresh the
`community_label` field in every card and community page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap: when invoked as `python scripts/ops/relabel_communities_llm.py`
# (i.e. without the package installed), add the repo's src/ to sys.path
# so `import nuthatch` resolves. Harmless when nuthatch is already
# importable (pipx, .venv, etc.).
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from nuthatch.clustering.relabel import (  # noqa: E402
    DEFAULT_RELABEL_BACKEND,
    DEFAULT_RELABEL_MODEL,
    build_default_invoke,
    discover_index_paths,
    relabel_communities,
)
from nuthatch.util import resolve_corpus  # noqa: E402


def _resolve_targets(kg_dir: Path, index_choice: str) -> list[Path]:
    """Map the --index choice to a list of paths to rewrite."""
    if index_choice == "all":
        return discover_index_paths(kg_dir)
    if index_choice == "canonical":
        canonical = kg_dir / "communities.json"
        return [canonical] if canonical.is_file() else []
    candidate = kg_dir / f"communities_{index_choice}.json"
    return [candidate] if candidate.is_file() else []


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0],
    )
    p.add_argument("--corpus", required=True, help="corpus name or path")
    p.add_argument(
        "--index", default="all",
        choices=("sbm", "leiden", "embeddings", "all", "canonical"),
        help="which file(s) to rewrite. 'all' covers canonical + every "
             "communities_<suffix>.json sibling.",
    )
    p.add_argument(
        "--backend", default=DEFAULT_RELABEL_BACKEND,
        help="LLM backend (ollama | openai | anthropic).",
    )
    p.add_argument(
        "--model", default=DEFAULT_RELABEL_MODEL,
        help=f"model id; default {DEFAULT_RELABEL_MODEL} (cheap local).",
    )
    p.add_argument(
        "--temperature", type=float, default=0.1,
        help="generation temperature; 0.1 keeps labels stable.",
    )
    args = p.parse_args(argv)

    corpus_root = resolve_corpus(args.corpus)
    kg_dir = corpus_root / ".kg"
    cards_dir = corpus_root / "cards"
    if not cards_dir.is_dir():
        print(
            f"[relabel] WARNING: no cards/ at {cards_dir}; label "
            "quality will suffer (titles unavailable).",
            file=sys.stderr,
        )

    targets = _resolve_targets(kg_dir, args.index)
    if not targets:
        print(f"[relabel] no matching files under {kg_dir}; nothing to do.")
        return 2

    print(f"[relabel] corpus: {corpus_root}")
    print(f"[relabel] LLM:    {args.backend} / {args.model}")
    invoke = build_default_invoke(
        backend=args.backend, model=args.model, temperature=args.temperature,
    )
    diffs = relabel_communities(targets, cards_dir, llm_invoke=invoke)

    for path, per_file in diffs.items():
        print(f"[relabel] {path.name}: {len(per_file)} communities")
        for cid, (old, new) in sorted(per_file.items()):
            print(f"  [{cid:>3}] {old!r}")
            print(f"        -> {new!r}")
        print(f"[relabel] wrote: {path}")

    print(
        "[relabel] done. Re-run `nuthatch render --corpus "
        f"{args.corpus}` to refresh card community_label fields."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
