# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Per-paper HTML companion renderer.

Purpose: produce a standalone HTML file alongside the markdown card,
    suitable for serving figures, equations (MathJax), and rich
    content the markdown form cannot express well. Lands under
    `<corpus>/html/<doc_id>.html`.

Inputs: `doc_id`, title, body HTML (typically Docling or Chandra's
    `.html` export), optional metadata dict for the head.

Outputs: an HTML5 string with a minimal stylesheet, MathJax loaded
    for inline + block LaTeX, and a footer linking back to the card.

Assumptions: caller has already produced body HTML (from one of the
    extractors); this module just wraps it in a consistent
    page-level template.
"""

from __future__ import annotations

import html
from typing import Any

_MATHJAX_CDN: str = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

_STYLE: str = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 80ch;
       margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }
header { border-bottom: 1px solid #ddd; padding-bottom: 0.5em; margin-bottom: 1em; }
.meta { color: #666; font-size: 0.9em; }
.meta-table { border-collapse: collapse; margin: 0.5em 0 1em; }
.meta-table td { padding: 0.1em 0.5em; }
.meta-table td:first-child { color: #888; }
footer { border-top: 1px solid #ddd; padding-top: 0.5em; margin-top: 2em;
         font-size: 0.85em; color: #888; }
""".strip()


def _meta_table(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    rows = []
    for key, value in metadata.items():
        rows.append(f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>")
    return f'<table class="meta-table">{"".join(rows)}</table>'


def render_html(
    *,
    doc_id: str,
    title: str,
    body_html: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Wrap an extractor's body HTML in a standalone page template."""
    metadata = metadata or {}
    safe_title = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{safe_title}</title>
<style>{_STYLE}</style>
<script id="MathJax-script" async src="{_MATHJAX_CDN}"></script>
</head>
<body>
<header>
<h1>{safe_title}</h1>
<div class="meta">doc_id: <code>{html.escape(doc_id)}</code></div>
{_meta_table(metadata)}
</header>
<main>
{body_html}
</main>
<footer>
Rendered by nuthatch. See the corresponding card under
<code>cards/{html.escape(doc_id)}.md</code> for Obsidian.
</footer>
</body>
</html>
"""
