# Clustering backends: comparative evaluation

Interpretation of the three-backend cluster quality eval run on 2026-05-27
against the 123-document `inputs` corpus. Raw eval reports live in
`pipeline_output/cluster_eval_report_*`. This document explains what
the numbers mean and why the backends diverge.

## Corpus and run parameters

- 123 documents ingested, embedded, and graphed.
- Graph: `inputs/.kg/graph/graph.json` (rebuilt 2026-05-27T06:51 UTC).
- Judge: `ollama::granite3-dense:8b` (IBM granite, non-reasoning; reliable
  structured JSON at local cost).
- Backends compared: `sbm_graph_tool` (principled), `leiden` (heuristic),
  `embeddings_kmeans` (embeddings-only).

## Summary comparison

| metric | SBM | Leiden | Embeddings |
| --- | ---: | ---: | ---: |
| communities | 5 | 4 | 7 |
| assigned docs | 123 | 123 | 123 |
| singletons | 0 | 1 | 1 |
| biggest community % | 26% | 46% | 28% |
| Surprise | 129.8 | 146.1 | 76.2 |
| Modularity Q | 0.114 | 0.197 | 0.127 |
| MDL (nats) | 3935 | n/a | n/a |
| LLM coherence | 0.40 | 0.50 | 0.65 |

## Metric interpretation guide

**Surprise** (Aldecoa-Marin) measures how unlikely the observed
intra-community edge density is under a random graph with the same total
edge count. It is the cross-backend comparable structural signal. Higher
means the partition captures real graph structure. It does not saturate on
dense or weighted graphs the way Modularity Q does.

**Modularity Q** is included for familiarity. On this graph Q stays below
0.20 across all backends; that is expected on dense or small graphs and
does not mean the partitions are bad. Read Surprise as the primary signal.

**MDL** (SBM only) is the description length of the fitted block model in
nats. It is the objective the SBM solver directly minimises. Lower means
the model compresses the graph more efficiently given the partition. It is
not comparable across backends or across different graphs.

**LLM coherence** is a soft signal: a judge reads five document titles per
community and rates thematic coherence on a 0-1 scale. It reflects
semantic legibility to a reader, not structural quality.

## Per-backend findings

### SBM (sbm_graph_tool, principled)

Surprise = 129.8. Five communities with no singletons, sizes 16-32
(mean 24.6, median 27). The partition is balanced: the largest community
covers only 26% of the corpus.

The SBM posterior is stable. Probing 500 MCMC steps around the MAP
estimate gives entropy std = 7.2 on a mean of 3954 nats (0.18% variation).
The MAP solution sits in a narrow optimum; it is not a fragile local
minimum.

The graph is structurally clique-like (global clustering coefficient 0.60)
with near-neutral degree assortativity (-0.01). Hubs do not preferentially
connect to other hubs or to periphery; the graph is relatively homogeneous.

The nested SBM found 8 hierarchy levels, but levels 1-7 all collapse to a
single block. The effective hierarchy is two levels: five leaf communities
merging into one corpus-wide super-community. At 123 documents the corpus
does not support a richer hierarchical structure; the SBM is reporting
that correctly.

LLM coherence = 0.40. The judge rated all five sampled communities
identically. Community labels were thematically plausible (viral evolution,
autonomous AI agents, multi-agent RL, single-cell RNA sequencing, AI
predictions in science) but the judge noted minor outliers in each. The
uniform 0.40 score suggests the judge applied a conservative rubric rather
than that the communities are semantically poor.

### Leiden (leiden, heuristic)

Surprise = 146.1 (highest of the three). Four communities; one singleton.

The partition has a hub pathology: the largest community contains 56 docs,
46% of the corpus. Leiden's modularity-maximising objective on this dense
graph produced one very large block alongside three smaller ones. High
Surprise here is partially a consequence of this concentration: one dense
block with many intra-community edges contributes heavily to the Surprise
score.

LLM coherence = 0.50. The three non-singleton communities were labelled
viral evolution and epistasis (56 docs), multi-agent RL (24 docs), and
multi-agent systems and AI agents (42 docs). The large "viral evolution"
community is almost certainly an over-merge of topically distinct documents
that share citation or co-mention edges.

### Embeddings k-means (embeddings_kmeans, embeddings-only)

Surprise = 76.2 (lowest). Seven communities; one singleton. The partition
ignores graph topology entirely and clusters documents by embedding
proximity alone.

LLM coherence = 0.65 (highest). Semantic signal is cleaner per cluster
because k-means in embedding space directly optimises for geometric
compactness in the semantic manifold. Community labels were specific:
gene regulation and chromatin loops, viral evolution and epistasis,
ML applications, agentic AI research, autonomous AI agents.

The lower Surprise reflects that the partition is not derived from the
graph. A partition that is semantically tight can still have low Surprise
if it does not align with the graph's edge structure.

## Why the backends diverge

SBM and Leiden both partition the graph; their objectives differ. Leiden
maximises modularity (fraction of intra-community edges above random
expectation). On dense graphs modularity is known to saturate and to favour
large blocks; the hub at 46% is a symptom of that. SBM instead minimises
description length: it asks how many bits are needed to encode the graph
given the partition. This regularises against over-large or over-small
communities and produces the most balanced result here.

Embeddings k-means does not see the graph at all. Its communities are
tight in semantic space but may split or merge graph-connected clusters
arbitrarily. The higher coherence score reflects this: thematic purity
is an explicit input to the algorithm, not an emergent property.

## Recommendation for the RAG use case

For community-aware retrieval, SBM is the recommended default. The
rationale:

- Balanced partition (no 46% hub). Community routing sends roughly equal
  fractions of the corpus to each block, so no one community dominates
  retrieval.
- Zero singletons. A singleton community produces a degenerate retrieval
  target.
- Stable MAP. Low posterior entropy variance means the partition is
  reproducible across re-runs.
- Rich structural metadata (MDL decomposition, hierarchy, assortativity)
  that can feed downstream LLM context.

Embeddings k-means is worth running as a complement when pure semantic
grouping is wanted (e.g. for surfacing thematically coherent topic pages),
but it should not replace the graph-backed partition for retrieval routing.

Leiden is not recommended at this corpus scale. The hub pathology is a
known limitation of modularity maximisation on small dense graphs. It may
improve as the corpus grows and the graph becomes sparser.

## Caveats

- Coherence scores are soft. Granite3-dense:8b is a capable but compact
  local model. Scores should be interpreted as relative comparisons within
  this run, not as absolute quality thresholds.
- The corpus is 123 documents: a pilot scale. Structural conclusions
  (especially the degenerate hierarchy) may shift as the corpus grows.
- The graph eval (Tier 1 for the graph itself, not the partition) is
  described in a separate document (`eval-graph.md`) once the post-fix
  rerun completes.
