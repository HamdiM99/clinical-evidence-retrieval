# ADR-004 · Re-ranker choice and serving

**Status:** accepted (M2), **revised** after the official SciFact annotations · evidence:
[reports/m2-reranking.md](../../reports/m2-reranking.md),
[article v2](../article/reranking-little-gain-real-risks.pdf)

## Context
After hybrid RRF (ADR-001), a cross-encoder can re-score the top fused candidates. Questions: which model,
how deep (`top_n`), trained how, served how, within the ≈ 100 ms re-ranking share of the p95 < 300 ms budget,
and whether it should be on at all.

## Decision
- **Default: no re-ranking.** The hybrid first stage alone is the default mode; re-ranking is a per-request /
  per-config flag.
- **Candidate model when enabled:** MiniLM-L6 cross-encoder fine-tuned with a **listwise** objective on SciFact
  train ("C2"), `top_n = 20` chosen on validation; documents below keep their fused order.
- **Serving when enabled:** GPU fp16 (measured on a T4); ONNX Runtime int8 as the CPU fallback.
- **Acceptance rule for any re-ranker:** a significant gain over the first stage on **evidence-aware** labels
  (SUPPORT / CONTRADICT), not only on BEIR relevance, plus a per-query regression review.

## Rationale (measured)
- No re-ranker beats hybrid RRF significantly on test: C2 +0.010 (p = 0.098) on NFCorpus, −0.001 (p = 0.952)
  on SciFact.
- The SciFact effects measured with BEIR labels are label artefacts: 38% of the BEIR relevant pairs contain no
  evidence. The off-the-shelf MiniLM loss (−0.028, p = 0.055) is −0.099 on the 112 claims without evidence and
  +0.013 (p = 0.414) on the 188 with evidence; C2's gain over MiniLM (+0.027, p = 0.002) is +0.054 vs +0.012
  (p = 0.203) on the same split. 39% of the training positives have no evidence either.
- Scored on evidence only, hybrid 0.855, MiniLM 0.867, C2 0.878: no significant difference.
- Re-rankers score relevance, not stance: one study keeps the same rank for a claim and its negation.
- Fine-tuning ablation: pointwise BCE and broad NFCorpus labels never beat the base model; listwise helps.
- Deeper re-ranking (`top_n` 50, 100) never helped and costs 2.5–5× more.
- Int8 ONNX: 4× smaller, ≈ 1.3–1.9× faster on CPU, same mean nDCG, but a different top-10 set on 38% of
  NFCorpus validation queries.

## Alternatives
- **Re-ranking on by default (previous decision):** rejected — added latency and risk without a measured gain.
- **bge-reranker-base (278M):** ≈ 3× slower than MiniLM on a T4 in fp16 (79 ms p95 at top 20, 197 ms at
  50: fits the budget only at top 20); not evaluated on quality yet.
- **LLM re-ranking:** incompatible with the latency budget.
- **Late interaction (ColBERT):** different index; not justified at this corpus size.

## Consequences
- Safety is not delegated to ranking: stance, negation, population and citation checks belong to a layer after
  generation.
- Evaluation must report evidence-aware metrics next to BEIR ones, and training labels must be filtered the same
  way (positives = documents that support or contradict the claim).
- If re-ranking is enabled: GPU fp16 (27 ms p95 at 20 passages), passage token ids stored at indexing time
  (tokenization is half of the GPU-path cost), models versioned with ranking-level regression tests.
- Each new model goes through the same protocol: validation selection, one test evaluation, paired permutation
  test and CIs against the current default.
