#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""
Script: audit_external_repo

Path: scripts/ops/audit_external_repo.py

Purpose: Generate a per-script audit markdown for a non-nuthatch repo
    (kestrel-latest, kestrel-v8, etc.) without requiring an inventory
    pre-generated in that repo. Walks `.py` files under a given root,
    parses each with `ast`, and emits a package-grouped markdown table
    with: path, module docstring (purpose paragraph), top-level
    classes / functions.

Inputs: positional `<repo_root>` and `<repo_label>` (used in the
    title and section headings).

Outputs: stdout (redirect to docs/architecture/<repo_label>.md).

Assumptions: target repo is a Python project. Non-Python sources are
    skipped. Reserved directories (`.git`, `__pycache__`, `node_modules`,
    `.venv`, `dist`, `build`) are excluded from the walk.

Parameters: --max-symbols (default 5) caps the classes+functions list
    per script so a 50-class file does not produce 50-row tables.

Failure Modes: a `.py` file that fails to parse (syntax error) is
    logged to stderr and emitted as `[unparseable]` in the table.

Author: Julen Gamboa

Created: 2026-05-26

Last Edited: 2026-05-26 by Julen Gamboa
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

_RESERVED = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build",
    ".tox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})


def _walk_py(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in _RESERVED for part in path.relative_to(root).parts):
            continue
        files.append(path)
    files.sort()
    return files


def _first_paragraph(text: str | None, limit: int = 240) -> str:
    if not text:
        return "_no module docstring_"
    # Prefer the `Purpose:` paragraph when present (matches nuthatch
    # script-docstring convention).
    for chunk in ("Purpose:", "purpose:"):
        if chunk in text:
            tail = text.split(chunk, 1)[1]
            para = tail.split("\n\n", 1)[0].strip()
            return _shorten(para, limit)
    para = text.strip().split("\n\n", 1)[0]
    return _shorten(para, limit)


def _shorten(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "..."


def _parse_one(path: Path) -> dict | None:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        print(f"WARN unparseable {path}: {exc}", file=sys.stderr)
        return None

    classes: list[str] = []
    functions: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                functions.append(node.name)

    return {
        "path": path,
        "docstring": ast.get_docstring(tree),
        "classes": classes,
        "functions": functions,
    }


def _bucket(rel: Path, root: Path) -> str:
    parts = rel.parts
    # Group by the first 1-2 segments depending on depth.
    if len(parts) == 1:
        return "_top"
    if len(parts) == 2:
        return parts[0]
    return f"{parts[0]}/{parts[1]}"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    p.add_argument("repo_root", type=Path)
    p.add_argument("repo_label", help="used in title + filenames (e.g. 'Kestrel (latest)')")
    p.add_argument("--max-symbols", type=int, default=5)
    args = p.parse_args(argv)

    root = args.repo_root.resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2

    py_files = _walk_py(root)
    by_pkg: dict[str, list[dict]] = defaultdict(list)
    for path in py_files:
        rel = path.relative_to(root)
        parsed = _parse_one(path)
        if parsed is None:
            parsed = {"path": path, "docstring": None, "classes": [], "functions": []}
        parsed["rel"] = rel
        by_pkg[_bucket(rel, root)].append(parsed)

    out: list[str] = []
    out.append("---")
    out.append(f'title: "{args.repo_label} per-script audit"')
    out.append("---")
    out.append("")
    out.append(f"Generated 2026-05-26 from `{root}`. Walked all `.py` files "
               f"outside reserved dirs ({', '.join(sorted(_RESERVED))}); "
               f"parsed each with `ast` to lift the module docstring and "
               f"top-level class / function declarations.")
    out.append("")
    out.append(f"**Counts.** {len(py_files)} Python files across "
               f"{len(by_pkg)} top-level packages.")
    out.append("")

    out.append("## Package overview")
    out.append("")
    out.append("| Package | Files |")
    out.append("| --- | ---: |")
    for pkg in sorted(by_pkg):
        label = "_top-level" if pkg == "_top" else pkg
        out.append(f"| `{label}` | {len(by_pkg[pkg])} |")
    out.append("")

    for pkg in sorted(by_pkg):
        label = "Top-level" if pkg == "_top" else f"`{pkg}`"
        out.append(f"## {label}")
        out.append("")
        out.append("| Path | Purpose | Key symbols |")
        out.append("| --- | --- | --- |")
        for entry in sorted(by_pkg[pkg], key=lambda e: str(e["rel"])):
            purpose = _first_paragraph(entry.get("docstring"))
            syms = (
                [f"`{c}`" for c in entry["classes"][:args.max_symbols]]
                + [f"`{f}()`" for f in entry["functions"][:args.max_symbols]]
            )[: args.max_symbols]
            sym_str = ", ".join(syms) or "_module-level only_"
            out.append(f"| `{entry['rel']}` | {purpose} | {sym_str} |")
        out.append("")

    sys.stdout.write("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
