# ADR-003 · Evaluation protocol

**Status:** accepted (week 1)

## Decision
- Public benchmarks with official relevance judgments (BEIR test splits), metrics at document level.
- Primary metric nDCG@10; Recall@100 to measure what reaches the re-ranker; MRR@10, MAP@100.
- Every run = (config YAML, index version, dataset) → TREC run file + metrics JSON + per-query CSV
  + latency percentiles.
- Two systems are compared with a paired permutation test on per-query nDCG@10 (10 000 permutations).
- Our metric implementation is cross-checked against `ranx` in the test suite.

## Rationale
Averages hide failures; significance tests prevent over-reading small differences on 50–300 queries;
storing runs makes every number in the README reproducible.
