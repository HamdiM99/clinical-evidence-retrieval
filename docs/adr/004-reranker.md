# ADR-004 · Re-ranker choice and serving

**Status:** accepted (M2) · evidence: [reports/m2-reranking.md](../../reports/m2-reranking.md)

## Context
After hybrid RRF (ADR-001), a cross-encoder can re-score the top fused candidates. Questions: which model,
how deep (`top_n`), trained how, served how, within the ≈ 100 ms re-ranking share of the p95 < 300 ms budget.

## Decision
- **Model:** MiniLM-L6 cross-encoder (22M parameters) fine-tuned with a **listwise** objective on in-domain
  claims (SciFact train), starting from `cross-encoder/ms-marco-MiniLM-L-6-v2` ("C2").
- **Depth:** `top_n = 20`, chosen on validation; documents below keep their fused order.
- **Serving:** GPU fp16 (measured on a T4); ONNX Runtime int8 as the CPU fallback.
- **Behind a flag:** re-ranking is optional per request/config, so the first stage alone stays a valid mode.

## Rationale (measured)
- Off-the-shelf MS MARCO MiniLM **hurts** SciFact (−0.029 nDCG@10 vs hybrid, p = 0.055); C2 removes that
  (+0.028 vs off-the-shelf, p = 0.002) and keeps NFCorpus (+0.001).
- Fine-tuning works only with clean in-domain labels and a ranking objective: pointwise BCE and the broad
  NFCorpus labels never beat the base model (ablation of learning rate, loss and data mix).
- Deeper re-ranking (`top_n` 50, 100) never helped and costs 2.5–5× more.
- Int8 ONNX: 4× smaller, ≈ 1.3–1.9× faster on CPU, no measured loss on validation.

## Alternatives
- **No re-ranker:** on the two test sets no re-ranker beats hybrid RRF significantly (NFCorpus +0.010,
  p ≈ 0.1; SciFact level). Kept as a supported mode; the re-ranker is retained for its MRR gain on NFCorpus,
  its robustness once fine-tuned, and as the place where better (clinician) labels will pay off.
- **bge-reranker-base (278M):** ≈ 3× slower than MiniLM on a T4 in fp16 (79 ms p95 at top 20, 197 ms at
  50: fits the budget only at top 20); not evaluated on quality yet.
- **LLM re-ranking:** incompatible with the latency budget.
- **Late interaction (ColBERT):** different index; not justified at this corpus size.

## Consequences
- Serving target is a GPU in fp16: C2 re-ranks 20 passages in 27 ms p95 on a T4 (63 ms at 50), within the
  ≈ 100 ms budget; the same work takes ≈ 1 s on a laptop CPU even in int8, so CPU int8 is a fallback, not the
  default.
- At that speed CPU tokenization is more than half of the cost: store passage token ids at indexing time.
- Labels are the main lever: a small, clean, in-domain set (e.g. clinician-validated guideline passages)
  is worth more than large weak labels.
- Each new model is accepted only through the same protocol: validation selection, one test evaluation,
  paired permutation test against the current default.
