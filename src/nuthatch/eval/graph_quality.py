#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Graph quality evaluation for a nuthatch corpus.

Loads the corpus's built graph (`<corpus>/.kg/graph.json`) and reports
structural-quality metrics. Optionally samples documents and asks an
LLM judge to rate the extracted entities' fidelity to the source.

Two trust tiers, mirroring `ragas_eval.py`:

Tier 1 - intrinsic (no LLM judge):
    Node count by EntityType, edge count by relation type, density,
    average degree, connected-component analysis, isolated-node count.
    Calibrated by construction; useful as a regression monitor across
    corpus growth + extractor changes.

Tier 2 - LLM-judged (optional, slow):
    For a sample of documents, ask the judge LLM "given this document's
    text, are the extracted entities present and correctly typed?"
    Returns per-entity precision (extracted entities that are real)
    and a rough recall ("did the extractor miss obvious entities?").

Usage:
    python scripts/eval/graph_eval.py \\
        --corpus inputs \\
        --judge-backend ollama --judge-model granite3-dense:8b \\
        --sample-docs 10

Skip LLM judging:
    python scripts/eval/graph_eval.py --corpus inputs --skip-judge

Output: pipeline_output/graph_eval_report_<UTC>.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from nuthatch.util import parse_llm_json, resolve_corpus, utc_tag
from nuthatch.util.llm_backends import build_llm


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0],
    )
    p.add_argument("--corpus", required=True, help="corpus name or path")
    p.add_argument("--sample-docs", type=int, default=100,
                   help="how many documents to sample for LLM-judged "
                        "extraction-quality scoring")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge-backend", default="ollama")
    p.add_argument("--judge-model", default="granite3-dense:8b",
                   help="LLM judge for extraction quality; structured-output "
                        "models (IBM Granite 3 etc.) follow the per-entity "
                        "JSON decomposition prompt reliably without burning "
                        "the token budget on a reasoning trace")
    p.add_argument("--judge-num-predict", type=int, default=1024)
    p.add_argument("--skip-judge", action="store_true",
                   help="run only intrinsic metrics; skip LLM-judged "
                        "extraction quality. Useful for fast smoke tests.")
    p.add_argument("--out-dir", default="pipeline_output")
    args = p.parse_args(argv)

    from nuthatch.corpus.layout import CorpusLayout
    from nuthatch.graph.io import load_graph

    corpus_root = resolve_corpus(args.corpus)
    layout = CorpusLayout(root=corpus_root)
    print(f"[graph-eval] corpus: {layout.root}")

    graph_path = layout.kg / "graph" / "graph.json"
    if not graph_path.exists():
        print(f"[graph-eval] no graph at {graph_path}; "
              "run `nuthatch graph` first.")
        return 2

    print(f"[graph-eval] loading graph: {graph_path}")
    g = load_graph(graph_path)
    print(f"[graph-eval] {g.number_of_nodes()} nodes, "
          f"{g.number_of_edges()} edges")

    print("[graph-eval] computing intrinsic metrics...")
    intrinsic = _compute_intrinsic(g)
    for k, v in intrinsic.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:24s} {vv}")
        else:
            print(f"  {k:24s} {v}")

    judge_results: list[dict] = []
    if not args.skip_judge:
        print(f"[graph-eval] LLM judge: "
              f"{args.judge_backend}::{args.judge_model}")
        judge_llm = build_llm(
            backend=args.judge_backend,
            model=args.judge_model,
            num_predict=args.judge_num_predict,
            temperature=0.0,
        )
        print(f"[graph-eval] sampling {args.sample_docs} docs for "
              "extraction-quality judging...")
        judge_results = _judge_extraction_quality(
            g, layout, judge_llm,
            sample_size=args.sample_docs, seed=args.seed,
        )
        if judge_results:
            mean_p = sum(r["precision"] for r in judge_results) / len(
                judge_results,
            )
            mean_r = sum(r["recall_proxy"] for r in judge_results) / len(
                judge_results,
            )
            print(f"  mean extraction precision: {mean_p:.3f}")
            print(f"  mean recall proxy:         {mean_r:.3f}")

    tag = utc_tag()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filename is self-describing: judge state inline so smoke (no-judge)
    # runs can be told from real ones without opening the file.
    judge_marker = "skipjudge" if args.skip_judge else "judge"
    report_path = out_dir / f"graph_eval_report_{judge_marker}_{tag}.md"
    _write_report(
        report_path=report_path,
        layout=layout,
        args=args,
        intrinsic=intrinsic,
        judge_results=judge_results,
    )
    print(f"[graph-eval] report: {report_path}")
    return 0

def _compute_intrinsic(g) -> dict:
    """Structural-quality metrics from the graph topology alone."""
    import networkx as nx

    n = g.number_of_nodes()
    m = g.number_of_edges()

    # Node-type breakdown via the `type` attribute that
    # nuthatch.graph.build sets on every node.
    type_counter: Counter = Counter()
    isolated = 0
    for node, data in g.nodes(data=True):
        type_counter[data.get("entity_type") or data.get("node_type", "unknown")] += 1
        # MultiDiGraph: treat as undirected for "isolated" definition.
        if g.degree(node) == 0:
            isolated += 1

    # Edge-relation breakdown via the `relation` attribute.
    relation_counter: Counter = Counter()
    for _, _, data in g.edges(data=True):
        relation_counter[data.get("relation", "unknown")] += 1

    # Density on the underlying simple undirected projection so the
    # MultiDiGraph multiplicity doesn't inflate the figure.
    undirected = nx.Graph(g.to_undirected(reciprocal=False))
    density = nx.density(undirected) if n > 1 else 0.0

    # Connected components on the undirected projection.
    components = list(nx.connected_components(undirected))
    n_components = len(components)
    largest_cc = max((len(c) for c in components), default=0)
    largest_cc_frac = largest_cc / n if n else 0.0

    # Average degree (sum of degrees / n, undirected).
    degrees = [d for _, d in undirected.degree()]
    avg_degree = sum(degrees) / n if n else 0.0
    max_degree = max(degrees, default=0)

    return {
        "node_count": n,
        "edge_count": m,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 3),
        "max_degree": max_degree,
        "isolated_nodes": isolated,
        "n_connected_components": n_components,
        "largest_component_size": largest_cc,
        "largest_component_fraction": round(largest_cc_frac, 3),
        "nodes_by_type": dict(type_counter.most_common()),
        "edges_by_relation": dict(relation_counter.most_common()),
    }


_EXTRACTION_JUDGE_PROMPT = """\
You are auditing an entity extractor's output for one document.

Given the document's title and a brief excerpt, decide for each \
extracted entity below:
  - "real": is this entity actually present in the document text?
  - "correctly_typed": does the entity type match what the text \
implies (e.g. "Smith J" labelled AUTHOR is correct if it's an author \
in the text; INCORRECT if it's actually a citation)?

Also estimate a recall proxy: scan the excerpt for any obvious \
entities (authors, citations, topics, methods, genes, organisations) \
that the extractor MISSED. Give a count, not a list.

DOCUMENT TITLE: {title}

DOCUMENT EXCERPT (first 800 chars of the body):
\"\"\"
{excerpt}
\"\"\"

EXTRACTED ENTITIES (type | name):
{entities}

Respond with valid JSON in exactly this shape:
{{
  "entities": [
    {{"name": "...", "real": true/false, "correctly_typed": true/false}},
    ...
  ],
  "missed_entity_count": <integer>
}}

Do not include code fences, prefix, or any text outside the JSON.
"""


def _judge_extraction_quality(
    g, layout, judge_llm, *, sample_size: int, seed: int,
) -> list[dict]:
    """Sample documents, ask the judge LLM to rate extracted entities.

    Returns a list of per-document scoring dicts with `precision`
    (fraction of extracted entities that are real AND correctly typed)
    and `recall_proxy` (1.0 - missed/(extracted+missed)).
    """
    import random

    rng = random.Random(seed)

    # Documents are nodes with type == "document". Pull their attached
    # entities via outgoing edges from the doc node.
    doc_nodes = [
        (node, data) for node, data in g.nodes(data=True)
        if data.get("node_type") == "document"
    ]
    if not doc_nodes:
        print("[graph-eval] no document nodes; skipping extraction judging")
        return []

    rng.shuffle(doc_nodes)
    sample = doc_nodes[:sample_size]

    results: list[dict] = []
    # extracted/ is always populated post-ingest; cards/ is only there
    # if `nuthatch render` has run. Reading from extracted/ avoids a
    # render dependency.
    extracted_dir = layout.extracted_dir
    for i, (doc_node, doc_data) in enumerate(sample, start=1):
        title = str(doc_data.get("title") or doc_data.get("doc_id") or doc_node)
        # Pull entities attached to this doc (neighbours of the doc node
        # that aren't themselves documents).
        entities = []
        for nbr in g.successors(doc_node):
            nbr_data = g.nodes[nbr]
            n_node_type = nbr_data.get("node_type", "")
            n_entity_type = nbr_data.get("entity_type") or "unknown"
            if n_node_type == "document":
                continue
            ntype = n_entity_type
            entities.append((ntype, str(nbr_data.get("name") or nbr)))
        if not entities:
            continue
        excerpt = _load_doc_excerpt(extracted_dir, str(doc_node))
        if not excerpt:
            continue

        entities_str = "\n".join(
            f"  - {etype} | {ename}" for etype, ename in entities[:30]
        )
        prompt = _EXTRACTION_JUDGE_PROMPT.format(
            title=title, excerpt=excerpt, entities=entities_str,
        )
        print(f"  [{i:>3d}/{len(sample)}] {title[:60]}...", flush=True)
        try:
            resp = judge_llm.invoke(prompt)
            content = getattr(resp, "content", str(resp)).strip()
            parsed = parse_llm_json(content)
        except Exception as exc:
            print(f"    WARN: judge call failed: {exc!s}")
            continue
        if parsed is None or "entities" not in parsed:
            continue

        # Defensive: judge sometimes returns a list-of-strings or
        # malformed entries instead of a list-of-dicts. Coerce and
        # filter so a single bad item doesn't crash the whole run.
        raw_entities = parsed.get("entities") or []
        if not isinstance(raw_entities, list):
            continue
        per_entity = [e for e in raw_entities if isinstance(e, dict)]
        real_and_typed = sum(
            1 for e in per_entity
            if e.get("real") and e.get("correctly_typed")
        )
        precision = (
            real_and_typed / len(per_entity) if per_entity else 0.0
        )
        missed = int(parsed.get("missed_entity_count", 0))
        denom = len(per_entity) + missed
        recall_proxy = (
            len(per_entity) / denom if denom else 0.0
        )
        results.append({
            "doc_id": str(doc_node),
            "title": title[:120],
            "n_extracted": len(per_entity),
            "n_real_and_typed": real_and_typed,
            "n_missed_estimate": missed,
            "precision": precision,
            "recall_proxy": recall_proxy,
        })
    return results


def _load_doc_excerpt(extracted_dir: Path, doc_id: str) -> str:
    """Best-effort load of a document excerpt from the extracted body.

    Graph node IDs are formatted `doc::<doc_id>`; the on-disk filename
    is `<doc_id>.md`. Strip the prefix before looking up.
    """
    bare_id = doc_id.split("::", 1)[1] if "::" in doc_id else doc_id
    candidates = [
        extracted_dir / f"{bare_id}.md",
        extracted_dir / f"{doc_id}.md",
        extracted_dir / f"{doc_id.replace(':', '_')}.md",
    ]
    for cand in candidates:
        if cand.is_file():
            text = cand.read_text(encoding="utf-8", errors="ignore")
            if text.startswith("---"):
                _, _, body = text.partition("---\n")[2].partition("\n---")
                text = body if body else text
            return text[:800]
    return ""


def _write_report(
    *,
    report_path: Path,
    layout,
    args: argparse.Namespace,
    intrinsic: dict,
    judge_results: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("---")
    lines.append(f'title: "Graph quality eval: {layout.root.name}"')
    lines.append(f"date: {datetime.now(UTC).isoformat()}")
    lines.append("---")
    lines.append("")
    lines.append("## How to read these numbers")
    lines.append("")
    lines.append(
        "Graph quality has two faces:",
    )
    lines.append("")
    lines.append(
        "- **Structural sanity (Tier 1)**: derived from the graph topology "
        "alone, no LLM judge. Tells you whether the graph has pathological "
        "shape (one giant component dominating, mostly isolated nodes, "
        "edge-type imbalance). Calibrated by construction.",
    )
    lines.append(
        "- **Extraction fidelity (Tier 2)**: LLM-judged precision on a "
        "sample of documents. Tells you whether the entities the extractor "
        "found are real and correctly typed. Includes a rough recall "
        "proxy based on the judge's count of obvious missed entities. "
        "Soft because LLM-judged.",
    )
    lines.append("")

    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Corpus: `{layout.root}`")
    lines.append(f"- Graph: `{layout.graph / 'graph.json'}`")
    if not args.skip_judge:
        lines.append(f"- Judge: `{args.judge_backend}::{args.judge_model}`")
        lines.append(f"- Documents sampled for judging: {args.sample_docs}")
    lines.append("")

    lines.append("## Tier 1: structural sanity")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    for k, v in intrinsic.items():
        if isinstance(v, dict):
            continue
        lines.append(f"| `{k}` | {v} |")
    lines.append("")

    if "nodes_by_type" in intrinsic:
        lines.append("### Nodes by type")
        lines.append("")
        lines.append("| type | count |")
        lines.append("| --- | ---: |")
        for t, c in intrinsic["nodes_by_type"].items():
            lines.append(f"| `{t}` | {c} |")
        lines.append("")

    if "edges_by_relation" in intrinsic:
        lines.append("### Edges by relation")
        lines.append("")
        lines.append("| relation | count |")
        lines.append("| --- | ---: |")
        for r, c in intrinsic["edges_by_relation"].items():
            lines.append(f"| `{r}` | {c} |")
        lines.append("")

    # Pathology sniff-tests
    lines.append("### Pathology sniff-tests")
    lines.append("")
    pathologies: list[str] = []
    if intrinsic.get("largest_component_fraction", 0) < 0.5:
        pathologies.append(
            f"- Fragmented: largest connected component covers only "
            f"{intrinsic['largest_component_fraction']:.0%} of nodes. "
            "Many documents may be unreachable from each other via "
            "shared entities."
        )
    if intrinsic.get("isolated_nodes", 0) > intrinsic.get("node_count", 1) * 0.1:
        pathologies.append(
            f"- Many isolates: {intrinsic['isolated_nodes']} nodes have "
            "degree 0. Extractor may be under-extracting per document."
        )
    if intrinsic.get("avg_degree", 0) < 1.0:
        pathologies.append(
            f"- Sparse: average degree {intrinsic['avg_degree']:.2f}. "
            "Most nodes are barely connected; the graph likely won't "
            "support meaningful community detection."
        )
    if not pathologies:
        pathologies.append("_No structural pathologies detected._")
    lines.extend(pathologies)
    lines.append("")

    if judge_results:
        mean_p = sum(r["precision"] for r in judge_results) / len(
            judge_results,
        )
        mean_r = sum(r["recall_proxy"] for r in judge_results) / len(
            judge_results,
        )
        lines.append("## Tier 2: LLM-judged extraction fidelity")
        lines.append("")
        lines.append(f"Sampled {len(judge_results)} documents. "
                     f"Judge: `{args.judge_backend}::{args.judge_model}`.")
        lines.append("")
        lines.append("| metric | value | reads as |")
        lines.append("| --- | ---: | --- |")
        lines.append(
            f"| `mean_extraction_precision` | {mean_p:.3f} | "
            "fraction of extracted entities that are real AND correctly typed |",
        )
        lines.append(
            f"| `mean_recall_proxy` | {mean_r:.3f} | "
            "n_extracted / (n_extracted + n_missed_estimate); higher = "
            "fewer obvious omissions |",
        )
        lines.append("")
        lines.append("### Per-document breakdown")
        lines.append("")
        lines.append("| doc | n_ext | n_real | n_missed | precision | recall_proxy |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for r in judge_results:
            lines.append(
                f"| {r['title']} | {r['n_extracted']} | "
                f"{r['n_real_and_typed']} | {r['n_missed_estimate']} | "
                f"{r['precision']:.3f} | {r['recall_proxy']:.3f} |"
            )
        lines.append("")
    else:
        lines.append("## Tier 2: LLM-judged extraction fidelity")
        lines.append("")
        lines.append("_skipped via `--skip-judge` or no usable samples._")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
