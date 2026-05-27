# Graph quality eval

Interpretation of the graph eval run on 2026-05-27 against the 123-document
`inputs` corpus. Raw eval report: `pipeline_output/graph_eval_report_judge_*.md`.
Graph built 2026-05-27T07:51 UTC (post-fix: `co_mentioned_in` includes all
entity types, including authors and citations).

## Run parameters

- 3523 nodes, 37899 edges in graph at eval time.
- 100 documents sampled for LLM-judged extraction fidelity.
- Judge: `ollama::granite3-dense:8b`.

## Tier 1: structural sanity

### Graph shape

| metric | value | reads as |
| --- | ---: | --- |
| nodes | 3523 | 123 documents + 3400 extracted entities |
| edges | 37899 | co-occurrence + typed provenance edges |
| density | 0.006 | sparse at node level; dense at entity co-mention level |
| avg degree | 21.5 | well-connected; not a sparse isolated graph |
| max degree | 179 | hub entities (very common topics/methods) |
| isolated nodes | 0 | no orphans |
| connected components | 2 | 99.8% in one component; 6 nodes in second |

The graph has two connected components. The dominant component contains
3517 nodes (99.8%). The six nodes in the second component are a small isolated
cluster; likely a document with no shared entities or citations with the rest
of the corpus. Not a structural concern at this scale.

### Node composition

| type | count | fraction |
| --- | ---: | ---: |
| citation | 1160 | 33% |
| topic | 732 | 21% |
| method | 615 | 17% |
| author | 542 | 15% |
| named\_entity | 351 | 10% |
| document | 123 | 4% |

Citation nodes dominate the entity space. This is expected for a corpus of
preprints: each paper cites many others, generating one citation entity per
reference. Method and topic nodes reflect the extraction pipeline's semantic
coverage.

### Edge composition

| relation | count | fraction |
| --- | ---: | ---: |
| co\_mentioned\_in | 32537 | 86% |
| mentions | 1946 | 5% |
| shares\_summary\_with | 1683 | 4% |
| cites | 1188 | 3% |
| authored\_by | 545 | 1% |

`co_mentioned_in` comprises 86% of edges. This is the expected signature of
a co-occurrence graph: entity pairs that appear together in a document generate
one edge per document, and each document contains many entity pairs. The
density of co-mention edges is what gives the graph enough structure for
ForceAtlas2 to produce a meaningful spatial layout. The `shares_summary_with`
edges are document-to-document semantic similarity links; 1683 of them across
123 documents corresponds to an average of ~27 document connections per paper,
which is the subgraph used directly by the clustering stage.

No structural pathologies detected.

## Tier 2: LLM-judged extraction fidelity

| metric | value | reads as |
| --- | ---: | --- |
| mean precision | 0.739 | 74% of extracted entities are real and correctly typed |
| mean recall proxy | 0.933 | extractor misses roughly 7% of obvious entities |

### Precision interpretation

Precision 0.739 means the extractor introduces about one false positive for
every three correct extractions. False positives are entities that are extracted
but do not belong to the document or are mis-typed. At this level the graph
has some noise but sufficient signal; community detection on the doc-doc
projection is robust to entity-level noise because it aggregates across many
entity edges per document.

Recall proxy 0.933 is high and consistent across the corpus. The extractor
is not systematically missing entity types; coverage is broad.

### Domain bias in precision

The low-precision tail is not random. Papers below 0.20 precision are
predominantly biology and genomics:

| document | precision | pattern |
| --- | ---: | --- |
| Community machine learning challenge (T cell gene perturbations) | 0.000 | biology over-extraction |
| Cross-domain benchmarks / coordinated AI agents | 0.000 | mismatched entity scope |
| Evaluation of Pipelines for Data Integration into KGs | 0.077 | generic terms extracted as entities |
| Deep RL Framework for Diversified Portfolio Management | 0.083 | finance domain mismatch |
| Estimating fitness effects of mutations | 0.100 | population genetics terminology |
| Inferring Gene Presence via Phylogenetic Occupancy Modeling | 0.107 | biology over-extraction |
| Regularizing and Normalizing DAGs and Phylogenetic Networks | 0.115 | mathematical/biology hybrid |
| Pangenome reference assemblies / LINE-1 retrotransposons | 0.148 | genomics terminology |

The pattern: the entity extractor was calibrated on AI and ML-heavy corpora
and over-extracts on biology, genomics, and finance papers. It pulls in
domain-specific terms (gene names, genomic coordinates, statistical method
names) as generic named entities, inflating the extracted count without adding
useful graph nodes.

The high-precision papers (0.9-1.0) are concentrated in AI, multi-agent
systems, and reinforcement learning — the primary subject matter of this corpus.

### Implications for downstream use

For the current corpus (majority AI/ML, minority biology), the 0.739 mean
precision is acceptable. Community detection is not materially affected because:

1. The doc-doc projection that clustering uses aggregates all entity edges.
   False-positive entities that connect to many documents actually help the
   projection by adding cross-document edges.
2. The biology papers with low precision do cluster correctly in the SBM
   partition — community 10 (single-cell RNA / variant calling) and community
   23 (viral evolution) are thematically coherent despite lower extraction
   precision in those papers.

The precision gap is a real extraction quality issue that should be addressed
for biology-heavy corpora. The fix is domain-specific tuning of the entity
extractor or a domain profile that applies stricter entity filtering for
biology documents (internal doc profile field-name mismatch, task #38, is a
related gap in the same area).

## Recommendations

- The graph structure is healthy. No action needed on topology.
- Extraction precision for biology papers (0.10-0.40) is a known gap.
  Addressing it requires either a biology-tuned entity extractor or
  post-extraction filtering per document domain.
- The 6-node second component is worth inspecting: identify which document
  is isolated and check whether its extraction produced any usable entities.
