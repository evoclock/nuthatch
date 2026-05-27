#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Cluster quality evaluation for a nuthatch corpus.

Loads the corpus's persisted community partition
(`<corpus>/.kg/communities.json`) and the underlying graph
(`<corpus>/.kg/graph.json`), then reports structural and
domain-coherence quality metrics.

Two trust tiers, mirroring `ragas_eval.py` + `graph_eval.py`:

Tier 1 - intrinsic (no LLM judge):
    n communities, size distribution (mean / median / IQR / max),
    modularity Q on the underlying graph, singleton fraction,
    giant-community check (no single community > 80% of corpus),
    coverage (fraction of doc nodes that received a community).

Tier 2 - LLM-judged (optional, slow):
    For a sample of communities, take N member-doc titles + brief
    summaries and ask the judge LLM "do these documents share a
    coherent theme? what is it?". Returns a per-community coherence
    score in [0, 1] and a generated label, plus the mean coherence
    across sampled communities.

Usage:
    python scripts/eval/cluster_eval.py \\
        --corpus inputs \\
        --judge-backend ollama --judge-model granite3-dense:8b \\
        --sample-communities 10 --docs-per-community 5

Skip LLM judging:
    python scripts/eval/cluster_eval.py --corpus inputs --skip-judge

Output: pipeline_output/cluster_eval_report_<UTC>.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from nuthatch.util import parse_llm_json, resolve_corpus, utc_tag
from nuthatch.util.llm_backends import build_llm


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0],
    )
    p.add_argument("--corpus", required=True, help="corpus name or path")
    p.add_argument(
        "--communities",
        default=None,
        help="suffix of the communities file to evaluate "
        "(e.g. 'sbm' loads communities_sbm.json). "
        "Omit for the default communities.json.",
    )
    p.add_argument(
        "--sample-communities",
        type=int,
        default=100,
        help="how many communities to sample for LLM-judged coherence scoring",
    )
    p.add_argument(
        "--docs-per-community",
        type=int,
        default=5,
        help="how many member docs to show the judge per community",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge-backend", default="ollama")
    p.add_argument(
        "--judge-model",
        default="granite3-dense:8b",
        help="LLM judge for community coherence; structured-output "
        "models (IBM Granite 3 etc.) follow the per-community "
        "JSON scoring rubric reliably without burning the token "
        "budget on a reasoning trace",
    )
    p.add_argument("--judge-num-predict", type=int, default=1024)
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="run only intrinsic metrics; skip LLM-judged "
        "coherence scoring. Useful for fast smoke tests.",
    )
    p.add_argument("--out-dir", default="pipeline_output")
    args = p.parse_args(argv)

    from nuthatch.clustering.persist import load_community_index
    from nuthatch.corpus.layout import CorpusLayout
    from nuthatch.graph.io import load_graph

    corpus_root = resolve_corpus(args.corpus)
    layout = CorpusLayout(root=corpus_root)
    print(f"[cluster-eval] corpus: {layout.root}")

    index_filename = f"communities_{args.communities}.json" if args.communities else None
    cidx = load_community_index(layout, index_filename=index_filename)
    if cidx is None:
        looked_at = layout.kg / (index_filename or "communities.json")
        print(
            f"[cluster-eval] no communities at {looked_at}; "
            "run `nuthatch cluster` first "
            "(or pass --communities <suffix> to pick a specific backend's output)."
        )
        return 2
    print(
        f"[cluster-eval] backend: {cidx.backend} (rigor: {cidx.rigor}); "
        f"{len(cidx.members)} communities; {len(cidx.flat)} doc "
        "assignments"
    )

    graph_path = layout.kg / "graph" / "graph.json"
    g = None
    if graph_path.exists():
        print(f"[cluster-eval] loading graph for modularity: {graph_path}")
        g = load_graph(graph_path)
        print(f"[cluster-eval] graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    else:
        print(f"[cluster-eval] WARNING: no graph at {graph_path}; modularity skipped")

    print("[cluster-eval] computing intrinsic metrics...")
    intrinsic = _compute_intrinsic(cidx, g)
    for k, v in intrinsic.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:24s} {vv}")
        else:
            print(f"  {k:24s} {v}")

    judge_results: list[dict] = []
    if not args.skip_judge:
        print(f"[cluster-eval] LLM judge: {args.judge_backend}::{args.judge_model}")
        judge_llm = build_llm(
            backend=args.judge_backend,
            model=args.judge_model,
            num_predict=args.judge_num_predict,
            temperature=0.0,
        )
        print(
            f"[cluster-eval] sampling {args.sample_communities} "
            "communities for coherence judging..."
        )
        judge_results = _judge_community_coherence(
            cidx,
            layout,
            judge_llm,
            sample_communities=args.sample_communities,
            docs_per_community=args.docs_per_community,
            seed=args.seed,
        )
        if judge_results:
            mean_c = sum(r["coherence_score"] for r in judge_results) / len(judge_results)
            print(f"  mean community coherence: {mean_c:.3f}")

    tag = utc_tag()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filename is self-describing: includes backend suffix (sbm /
    # leiden / embeddings) so backend comparisons are obvious on disk,
    # plus judge state (judge / skipjudge) to flag smoke runs.
    backend_marker = args.communities or cidx.backend or "default"
    judge_marker = "skipjudge" if args.skip_judge else "judge"
    report_path = out_dir / (f"cluster_eval_report_{backend_marker}_{judge_marker}_{tag}.md")
    _write_report(
        report_path=report_path,
        layout=layout,
        args=args,
        cidx=cidx,
        intrinsic=intrinsic,
        judge_results=judge_results,
    )
    print(f"[cluster-eval] report: {report_path}")
    return 0


def _compute_intrinsic(cidx, g) -> dict:
    """Structural-quality metrics from the partition + graph."""
    sizes = [len(members) for members in cidx.members.values()]
    if not sizes:
        return {"n_communities": 0}

    sizes_sorted = sorted(sizes)
    n_communities = len(sizes)
    n_assigned = len(cidx.flat)
    n_singletons = sum(1 for s in sizes if s == 1)
    biggest = max(sizes)
    smallest = min(sizes)
    mean_size = sum(sizes) / n_communities
    median_size = sizes_sorted[n_communities // 2]
    q1 = sizes_sorted[n_communities // 4]
    q3 = sizes_sorted[(3 * n_communities) // 4]

    biggest_fraction = biggest / n_assigned if n_assigned else 0.0
    singleton_fraction = n_singletons / n_communities

    gt = getattr(cidx, "gt_metrics", None) or {}

    # For SBM runs use graph-tool's own Q; for other backends fall back
    # to the networkx computation so the cross-backend comparison row
    # is still populated.
    modularity: float | None
    if gt.get("gt_modularity") is not None:
        modularity = gt["gt_modularity"]
    elif g is not None:
        modularity = _modularity(cidx, g)
    else:
        modularity = None

    surprise = _surprise(cidx, g) if g is not None else None

    n_nodes = n_assigned or 1
    mdl_nats = cidx.mdl_nats if hasattr(cidx, "mdl_nats") else None
    mdl_nats_per_node = round(mdl_nats / n_nodes, 4) if mdl_nats is not None else None

    return {
        "n_communities": n_communities,
        "n_assigned": n_assigned,
        "n_singletons": n_singletons,
        "singleton_fraction": round(singleton_fraction, 3),
        "biggest_community_size": biggest,
        "biggest_community_fraction": round(biggest_fraction, 3),
        "smallest_community_size": smallest,
        "mean_community_size": round(mean_size, 2),
        "median_community_size": median_size,
        "size_q1": q1,
        "size_q3": q3,
        "modularity_Q": (round(modularity, 4) if modularity is not None else None),
        "surprise": (round(surprise, 4) if surprise is not None else None),
        "mdl_nats": (round(mdl_nats, 2) if mdl_nats is not None else None),
        "mdl_nats_per_node": mdl_nats_per_node,
        "backend": cidx.backend,
        "rigor": cidx.rigor,
        "n_levels": cidx.n_levels,
        # SBM-only fields; None for non-SBM backends
        "_gt": gt,
    }


def _surprise(cidx, g) -> float | None:
    """Compute Aldecoa-Marin Surprise on the doc-doc projection.

    Surprise S = -log10 P(X >= m_P) where X ~ Hypergeom(M, F, m):
      M  = n*(n-1)/2  total possible edges
      F  = sum_c n_c*(n_c-1)/2  total possible intra-community edges
      m  = observed edges
      m_P = observed intra-community edges

    Measures how unlikely it is to see at least m_P intra-community
    edges by chance in a random graph with the same (n, m). Higher
    is better; no upper bound. Does not saturate on dense graphs and
    requires no resolution parameter.

    Reference: Aldecoa & Marin (2011) Sci Rep 1:51.
    """
    try:
        from scipy.stats import hypergeom
    except ImportError:
        return None

    from collections import defaultdict

    from nuthatch.clustering.projection import project_to_doc_doc

    proj = project_to_doc_doc(g)
    n = proj.number_of_nodes()
    m = proj.number_of_edges()
    if n < 2 or m == 0:
        return None

    # Map each projection node to its community, tolerating either
    # the `doc::` prefixed form or the bare form in cidx.flat.
    node_to_comm: dict[str, int] = {}
    for node in proj.nodes():
        bare = node.split("::", 1)[1] if "::" in node else node
        cid = cidx.flat.get(node)
        if cid is None:
            cid = cidx.flat.get(bare)
        if cid is None:
            return None
        node_to_comm[node] = cid

    # Group nodes by community.
    communities: dict[int, set[str]] = defaultdict(set)
    for node, cid in node_to_comm.items():
        communities[cid].add(node)

    # M = total possible edges; F = sum of per-community pair counts.
    M = n * (n - 1) // 2
    F = sum(
        len(members) * (len(members) - 1) // 2
        for members in communities.values()
        if len(members) >= 2
    )
    if F == 0:
        return None

    # m_P = total intra-community edges observed.
    m_P = sum(1 for u, v in proj.edges() if node_to_comm.get(u) == node_to_comm.get(v))
    if m_P == 0:
        return 0.0

    try:
        # logsf returns log(P(X >= m_P)); flip sign for Surprise.
        # Convert from nats to log10 for a readable positive number.
        import math

        log_p = hypergeom.logsf(m_P - 1, M, F, m)
        return float(-log_p / math.log(10))
    except Exception:
        return None


def _modularity(cidx, g) -> float | None:
    """Compute weighted modularity Q against the persisted partition.

    Modularity must be measured on the SAME graph the clustering ran on.
    SBM and Leiden cluster the doc-doc bipartite projection (built by
    `nuthatch.clustering.projection.project_to_doc_doc`), not the raw
    augmented graph. Measuring Q on the augmented graph's doc-induced
    subgraph misses 2071 weighted edges that the projection adds via
    shared-entity bipartite collapse — Q comes out artefactually low.

    Also passes `weight="weight"` so the projection's 1-to-20 weight
    range contributes correctly; without it, networkx treats every
    edge as unit weight and a 0.9-similarity edge gets the same
    contribution as a 1-shared-entity edge.

    For the embeddings backend (which clusters in vector space rather
    than on a graph), we still measure Q on the projection — it's the
    closest "what would the graph version of this partition look
    like?" reference, which is what we want to compare across backends.
    """
    import networkx as nx

    from nuthatch.clustering.projection import project_to_doc_doc

    proj = project_to_doc_doc(g)
    if proj.number_of_edges() == 0:
        return None

    # Build the communities-as-sets structure networkx expects.
    by_comm: dict[int, set[str]] = {}
    for doc_id, cid in cidx.flat.items():
        if doc_id in proj.nodes:
            by_comm.setdefault(cid, set()).add(doc_id)
    communities = list(by_comm.values())
    if not communities:
        return None
    try:
        return float(nx.community.modularity(proj, communities, weight="weight"))
    except Exception:
        return None


_COHERENCE_PROMPT = """\
You are auditing a community-detection partition over a research \
corpus. Given a community's member documents (titles + short \
summaries), decide whether they share a coherent theme.

A coherent community is one where the documents could plausibly be \
grouped together by a domain expert reading them. Incoherent \
communities mix unrelated topics.

DOCUMENTS IN THIS COMMUNITY:
{docs}

Respond with valid JSON in exactly this shape:
{{
  "coherence_score": <float 0.0 to 1.0>,
  "theme_label": "<short label, 2-6 words>",
  "rationale": "<one sentence>"
}}

Scoring rubric:
  1.0  All documents share a tight, specific theme.
  0.7  Most documents share a broader theme; minor outliers.
  0.4  Two or three sub-themes coexist; partition is loose.
  0.0  No discernible shared theme; documents look unrelated.

If the community is too small (< 2 documents), score 0.0 with \
theme_label "singleton".

Do not include code fences, prefix, or any text outside the JSON.
"""


def _judge_community_coherence(
    cidx,
    layout,
    judge_llm,
    *,
    sample_communities: int,
    docs_per_community: int,
    seed: int,
) -> list[dict]:
    """Sample communities, ask the judge LLM to score coherence."""
    import random

    rng = random.Random(seed)

    # Score every community that has >=2 members; sample if too many.
    candidate_cids = [cid for cid, members in cidx.members.items() if len(members) >= 2]
    if not candidate_cids:
        print("[cluster-eval] no communities with >=2 members; skipping coherence judging")
        return []

    rng.shuffle(candidate_cids)
    sampled = candidate_cids[:sample_communities]

    # extracted/ is populated at ingest; cards/ requires `nuthatch render`.
    # Read from extracted/ so cluster eval doesn't depend on render.
    extracted_dir = layout.extracted_dir
    results: list[dict] = []
    for i, cid in enumerate(sampled, start=1):
        members = cidx.members.get(cid, [])
        # Filter to document nodes only. Graph-based backends (SBM,
        # Leiden) partition the FULL augmented graph including entity
        # nodes (`author::...`, `citation::...`, `topic::...`,
        # `method::...`, `named_entity::...`). Entity nodes have no
        # extracted-body card, so sampling them and asking the judge
        # for "coherence" produces nonsense rationales ("only names,
        # no content, cannot judge"). The embeddings backend doesn't
        # hit this because it only clusters document centroids.
        # Filter is by `doc::` prefix; bare ids (no prefix) are
        # treated as documents for backwards compat with backends
        # that don't namespace.
        doc_members = [m for m in members if str(m).startswith("doc::") or "::" not in str(m)]
        n_doc_members = len(doc_members)
        sample_members = doc_members[:docs_per_community]

        doc_blurbs: list[str] = []
        for doc_id in sample_members:
            title, summary = _load_doc_title_and_summary(extracted_dir, doc_id)
            doc_blurbs.append(
                f"- **{title}**\n  {summary[:300]}",
            )
        if not doc_blurbs:
            print(
                f"  [{i:>3d}/{len(sampled)}] community {cid} "
                f"({len(members)} members, 0 docs) - skip: "
                "no document nodes in this community",
                flush=True,
            )
            continue
        docs_str = "\n".join(doc_blurbs)
        prompt = _COHERENCE_PROMPT.format(docs=docs_str)
        print(
            f"  [{i:>3d}/{len(sampled)}] community {cid} "
            f"({len(members)} members, {n_doc_members} docs)...",
            flush=True,
        )
        try:
            resp = judge_llm.invoke(prompt)
            content = getattr(resp, "content", str(resp)).strip()
            parsed = parse_llm_json(content)
        except Exception as exc:
            print(f"    WARN: judge call failed: {exc!s}")
            continue
        if parsed is None or "coherence_score" not in parsed:
            continue
        try:
            score = float(parsed["coherence_score"])
        except (TypeError, ValueError):
            continue
        results.append(
            {
                "community_id": int(cid),
                "n_members": len(members),
                "n_doc_members": n_doc_members,
                "n_judged": len(sample_members),
                "coherence_score": score,
                "theme_label": str(parsed.get("theme_label", "")).strip(),
                "rationale": str(parsed.get("rationale", "")).strip(),
            }
        )
    return results


def _load_doc_title_and_summary(
    extracted_dir: Path,
    doc_id: str,
) -> tuple[str, str]:
    """Best-effort load of (title, summary) from the extracted body.

    Document IDs in communities.json are bare (e.g. "2605.22305v1"),
    matching the extracted/<doc_id>.md filename. Title comes from the
    sibling .meta.json; summary is the first 500 chars of the body.
    """
    import json as _json

    bare_id = doc_id.split("::", 1)[1] if "::" in doc_id else doc_id
    title = bare_id
    summary = ""
    body_path = extracted_dir / f"{bare_id}.md"
    if body_path.is_file():
        text = body_path.read_text(encoding="utf-8", errors="ignore")
        if text.startswith("---"):
            _, _, rest = text.partition("---\n")
            fm, _, body = rest.partition("\n---")
            for line in fm.splitlines():
                if line.strip().startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"').strip("'")
                    break
            summary = body.strip()[:500]
        else:
            summary = text.strip()[:500]
    meta_path = extracted_dir / f"{bare_id}.meta.json"
    if meta_path.is_file():
        try:
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            mt = meta.get("metadata", {}).get("title")
            if mt:
                title = str(mt)
        except Exception:
            pass
    return title, summary


def _write_report(
    *,
    report_path: Path,
    layout,
    args: argparse.Namespace,
    cidx,
    intrinsic: dict,
    judge_results: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("---")
    lines.append(f'title: "Cluster quality eval: {layout.root.name}"')
    lines.append(f"date: {datetime.now(UTC).isoformat()}")
    lines.append("---")
    lines.append("")
    lines.append("## How to read these numbers")
    lines.append("")
    lines.append(
        "- **Structural sanity (Tier 1)**: derived from the partition + "
        "graph topology, no LLM judge. Reports three complementary "
        "quality signals: Surprise (density vs random baseline; higher "
        "is better), Modularity Q (familiar benchmark; saturates on "
        "dense graphs), and MDL in nats (SBM only; the objective "
        "function directly optimised; lower is better). See the "
        "quality-metrics section for interpretation guidance.",
    )
    lines.append(
        "- **Domain coherence (Tier 2)**: LLM-judged thematic coherence "
        "on a sample of communities. Tells you whether the discovered "
        "groupings make sense to a reader. Soft, since LLM-judged.",
    )
    lines.append("")

    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Corpus: `{layout.root}`")
    lines.append(f"- Communities: `{layout.kg / 'communities.json'}`")
    lines.append(f"- Backend: `{cidx.backend}` (rigor: `{cidx.rigor}`)")
    if not args.skip_judge:
        lines.append(f"- Judge: `{args.judge_backend}::{args.judge_model}`")
        lines.append(f"- Communities sampled: {args.sample_communities}")
        lines.append(f"- Docs per community shown to judge: {args.docs_per_community}")
    lines.append("")

    lines.append("## Tier 1: structural sanity")
    lines.append("")
    lines.append("### Partition shape")
    lines.append("")
    shape_keys = {
        "n_communities",
        "n_assigned",
        "n_singletons",
        "singleton_fraction",
        "biggest_community_size",
        "biggest_community_fraction",
        "smallest_community_size",
        "mean_community_size",
        "median_community_size",
        "size_q1",
        "size_q3",
        "backend",
        "rigor",
        "n_levels",
    }
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    for k, v in intrinsic.items():
        if k in shape_keys and not k.startswith("_"):
            lines.append(f"| `{k}` | {v} |")
    lines.append("")

    lines.append("### Partition quality metrics")
    lines.append("")
    lines.append(
        "Three complementary quality signals. Read together, not in isolation:",
    )
    lines.append("")
    lines.append(
        "- **Surprise** (Aldecoa-Marin): how unlikely is the observed "
        "intra-community density under a random graph with the same edge "
        "count? Higher = more structured partition. No upper bound; "
        "compare across backends on the same graph. Does not saturate "
        "on dense weighted graphs.",
    )
    lines.append(
        "- **Modularity Q** (Newman-Girvan): fraction of intra-community "
        "edges above random expectation, weighted by edge weight. "
        "Familiar benchmark; >0.3 indicates real structure, >0.5 is "
        "strong. Mathematically caps below ~0.5 on dense or small graphs "
        "regardless of partition quality — use Surprise as the primary "
        "signal when Q looks low.",
    )
    lines.append(
        "- **MDL (nats)** (SBM only): description length of the fitted "
        "block model. The value minimize_nested_blockmodel_dl directly "
        "optimises. Lower = more compressive; the model needs fewer bits "
        "to encode the graph given the partition. Useful for comparing "
        "SBM runs on the same graph (e.g. nested vs flat). Not "
        "comparable across different graphs or backends.",
    )
    lines.append("")
    lines.append("| metric | value | reads as |")
    lines.append("| --- | ---: | --- |")
    q = intrinsic.get("modularity_Q")
    s = intrinsic.get("surprise")
    m = intrinsic.get("mdl_nats")
    mpn = intrinsic.get("mdl_nats_per_node")
    lines.append(
        f"| `surprise` | {s if s is not None else 'n/a'} | "
        "higher = partition more structured than random (Surprise) |",
    )
    lines.append(
        f"| `modularity_Q` | {q if q is not None else 'n/a'} | "
        ">0.3 real structure; saturates on dense graphs |",
    )
    lines.append(
        f"| `mdl_nats` | {m if m is not None else 'n/a (non-SBM)'} | "
        "lower = more compressive model (SBM only) |",
    )
    lines.append(
        f"| `mdl_nats_per_node` | "
        f"{mpn if mpn is not None else 'n/a (non-SBM)'} | "
        "MDL normalised by doc count; comparable across corpus sizes |",
    )
    lines.append("")

    # Pathology sniff-tests
    lines.append("### Pathology sniff-tests")
    lines.append("")
    pathologies: list[str] = []
    biggest_frac = intrinsic.get("biggest_community_fraction", 0)
    if biggest_frac > 0.8:
        pathologies.append(
            f"- Giant community: largest community covers "
            f"{biggest_frac:.0%} of assigned nodes. Clustering may have "
            "collapsed; check the resolution parameter or backend choice.",
        )
    elif biggest_frac > 0.3:
        pathologies.append(
            f"- Hub community: largest community covers "
            f"{biggest_frac:.0%} of assigned nodes. The partition is "
            "imbalanced; one block dominates. Often a sign that the "
            "graph has a densely-connected core or that the resolution "
            "needs tuning.",
        )
    if intrinsic.get("singleton_fraction", 0) > 0.5:
        pathologies.append(
            f"- Mostly singletons: {intrinsic['singleton_fraction']:.0%} "
            "of communities have a single member. The graph may be too "
            "sparse to support community detection, or the resolution is "
            "too fine.",
        )
    mod = intrinsic.get("modularity_Q")
    surp = intrinsic.get("surprise")
    if mod is not None and mod < 0.3 and (surp is None or surp < 5.0):
        # Only flag low Q as a pathology if Surprise also looks weak.
        # Low Q alone on a dense weighted graph is a known mathematical
        # artefact (resolution limit), not a partition problem.
        pathologies.append(
            f"- Low partition quality: Q = {mod:.3f}, "
            f"Surprise = {surp if surp is not None else 'n/a'}. "
            "Both signals are weak; the underlying graph may lack "
            "community structure, or the partition is wrong for it.",
        )
    elif mod is not None and mod < 0.3 and surp is not None and surp >= 5.0:
        pathologies.append(
            f"- Low modularity Q = {mod:.3f} (but Surprise = {surp:.2f} "
            "is positive). Q is known to saturate on dense weighted "
            "graphs; Surprise is the more reliable signal here. "
            "Partition likely has real structure.",
        )
    if not pathologies:
        pathologies.append("_No structural pathologies detected._")
    lines.extend(pathologies)
    lines.append("")

    gt = intrinsic.get("_gt") or {}
    if gt:
        lines.append("## SBM-specific metrics (graph-tool)")
        lines.append("")
        lines.append(
            "Metrics available only for the principled SBM backend. "
            "Not directly comparable to Leiden or embeddings outputs; "
            "these expose the internal model structure that the other "
            "backends do not fit.",
        )
        lines.append("")

        lines.append("### MDL decomposition")
        lines.append("")
        lines.append(
            "Full description length = likelihood term + encoding overhead. "
            "A lower likelihood term means the partition explains the edges well. "
            "A lower encoding overhead means the model structure itself is simple.",
        )
        lines.append("")
        lines.append("| term | nats | reads as |")
        lines.append("| --- | ---: | --- |")
        _mdl_total = gt.get("mdl_nats")
        _mdl_like = gt.get("mdl_likelihood_nats")
        _mdl_oh = gt.get("mdl_encoding_overhead")
        lines.append(
            f"| `mdl_total` | "
            f"{round(_mdl_total, 2) if _mdl_total is not None else 'n/a'} | "
            "full description length (state.entropy()) |",
        )
        lines.append(
            f"| `mdl_likelihood` | "
            f"{round(_mdl_like, 2) if _mdl_like is not None else 'n/a'} | "
            "how well edges are explained by the partition (dl=False) |",
        )
        lines.append(
            f"| `mdl_encoding_overhead` | "
            f"{_mdl_oh if _mdl_oh is not None else 'n/a'} | "
            "cost of encoding the model structure; lower = simpler model |",
        )
        lines.append("")

        mdl_per_level = gt.get("mdl_per_level")
        blocks_per_level = gt.get("blocks_per_level")
        if mdl_per_level or blocks_per_level:
            lines.append("### Hierarchy breakdown")
            lines.append("")
            lines.append(
                "How description length and block count distribute across the "
                "nested SBM hierarchy. Level 0 is the finest (leaf) partition.",
            )
            lines.append("")
            lines.append("| level | non-empty blocks | entropy (nats) |")
            lines.append("| --- | ---: | ---: |")
            n_lv = max(
                len(mdl_per_level) if mdl_per_level else 0,
                len(blocks_per_level) if blocks_per_level else 0,
            )
            for i in range(n_lv):
                blk = (
                    blocks_per_level[i] if blocks_per_level and i < len(blocks_per_level) else "n/a"
                )
                ent = (
                    round(mdl_per_level[i], 2)
                    if mdl_per_level and i < len(mdl_per_level)
                    else "n/a"
                )
                label = f"{i} (leaf)" if i == 0 else str(i)
                lines.append(f"| {label} | {blk} | {ent} |")
            lines.append("")

        lines.append("### Graph structural properties")
        lines.append("")
        lines.append("| metric | value | reads as |")
        lines.append("| --- | ---: | --- |")
        _assort = gt.get("degree_assortativity")
        _clust = gt.get("global_clustering")
        _gtq = gt.get("gt_modularity")
        lines.append(
            f"| `gt_modularity` | "
            f"{round(_gtq, 4) if _gtq is not None else 'n/a'} | "
            "graph-tool Q; consistent with the graph the SBM fitted |",
        )
        lines.append(
            f"| `degree_assortativity` | "
            f"{round(_assort, 4) if _assort is not None else 'n/a'} | "
            ">0 hubs connect to hubs; <0 hubs connect to periphery |",
        )
        lines.append(
            f"| `global_clustering` | "
            f"{round(_clust, 4) if _clust is not None else 'n/a'} | "
            "triangle density; high = clique-like; low = hub-spoke |",
        )
        lines.append("")

        _pe_mean = gt.get("posterior_entropy_mean")
        _pe_std = gt.get("posterior_entropy_std")
        if _pe_mean is not None or _pe_std is not None:
            lines.append("### Posterior uncertainty (MCMC probe)")
            lines.append("")
            lines.append(
                "Entropy sampled over 500 MCMC steps on a copy of the MAP "
                "state. Low std = MAP is a stable optimum; high std = "
                "rugged posterior landscape, partition less certain.",
            )
            lines.append("")
            lines.append("| metric | value |")
            lines.append("| --- | ---: |")
            lines.append(
                f"| `posterior_entropy_mean` | {_pe_mean if _pe_mean is not None else 'n/a'} |",
            )
            lines.append(
                f"| `posterior_entropy_std` | {_pe_std if _pe_std is not None else 'n/a'} |",
            )
            lines.append("")

    if judge_results:
        mean_c = sum(r["coherence_score"] for r in judge_results) / len(judge_results)
        lines.append("## Tier 2: LLM-judged domain coherence")
        lines.append("")
        lines.append(
            f"Sampled {len(judge_results)} communities. Judge: "
            f"`{args.judge_backend}::{args.judge_model}`.",
        )
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | ---: |")
        lines.append(f"| `mean_coherence` | {mean_c:.3f} |")
        lines.append("")
        lines.append("### Per-community detail")
        lines.append("")
        lines.append(
            "| community | size | judged | coherence | label | rationale |",
        )
        lines.append("| --- | ---: | ---: | ---: | --- | --- |")
        for r in judge_results:
            lines.append(
                f"| {r['community_id']} | {r['n_members']} | "
                f"{r['n_judged']} | {r['coherence_score']:.2f} | "
                f"{r['theme_label']} | {r['rationale']} |"
            )
        lines.append("")
    else:
        lines.append("## Tier 2: LLM-judged domain coherence")
        lines.append("")
        lines.append("_skipped via `--skip-judge` or no usable samples._")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
