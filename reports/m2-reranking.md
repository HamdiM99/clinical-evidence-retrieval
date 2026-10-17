# M2 — Re-ranking

Re-ranking is replayed offline on exported first-stage candidates (`cer candidates`, then `cer rerank`):
every re-ranker sees exactly the same hybrid RRF top 100, scores are cached once and each `top_n` is
derived from the cache. Replaying the test candidates without re-ranking reproduces the M1 hybrid runs
exactly (same rankings, same metrics).

Hardware: laptop CPU (Intel i7-8565U, 4 cores, no GPU). Passages are the best chunk of each document with
its title, ≈ 220 words, truncated with the query to 256 tokens.

## Off-the-shelf baseline: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Choice of top_n (tuning sets, never test)

Rule fixed before looking at the numbers: best mean nDCG@10 over NFCorpus dev (324 queries) and the first
200 SciFact train queries (sorted by id; excluded from fine-tuning data); a smaller top_n wins if within 0.005.

| tuning set | hybrid | top 20 | top 50 | top 100 |
|---|---|---|---|---|
| NFCorpus dev | 0.3222 | **0.3308** | 0.3286 | 0.3276 |
| SciFact train[:200] | 0.7497 | **0.7301** | 0.7232 | 0.7216 |
| mean | 0.5360 | **0.5304** | 0.5259 | 0.5246 |
| re-ranking p50 (CPU) | – | ≈ 1.1–1.3 s | ≈ 2.8–3.3 s | ≈ 5.5–6.5 s |

Selected: **top_n = 20**. Re-ranking deeper never helps this model: the extra candidates it promotes are
mostly wrong.

### Test (reported once)

| dataset | system | nDCG@10 | MRR@10 | MAP@100 | Recall@100 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| NFCorpus | hybrid RRF | 0.3528 | 0.5596 | 0.1629 | 0.3013 | 127 | 143 |
| NFCorpus | + MiniLM top 20 | 0.3615 | 0.5877 | 0.1690 | 0.3013 | 1263 | 1580 |
| SciFact | hybrid RRF | 0.7298 | 0.6970 | 0.6900 | 0.9483 | 150 | 189 |
| SciFact | + MiniLM top 20 | 0.7013 | 0.6721 | 0.6649 | 0.9483 | 1316 | 1601 |

Latency is end to end per query (first stage measured at export + re-ranking measured on this CPU).
Recall@100 is unchanged by construction (consistency check).

Paired permutation test vs hybrid RRF (10,000 permutations):

| dataset | Δ nDCG@10 | p | Δ MRR@10 | p | Δ MAP@100 | p | per query (win / loss / tie) |
|---|---|---|---|---|---|---|---|
| NFCorpus | +0.009 | 0.157 | +0.028 | 0.067 | +0.006 | 0.063 | 106 / 91 / 126 |
| SciFact | −0.029 | 0.055 | −0.025 | 0.178 | −0.025 | 0.153 | 48 / 62 / 190 |

### Reading

- **No significant gain from an off-the-shelf MS MARCO cross-encoder on top of a strong hybrid.** On
  NFCorpus it is a small, non-significant improvement (better MRR: the first relevant document moves up);
  on SciFact it is a borderline-significant **loss**. Same direction on the tuning sets and on test.
- This matches the literature once the baseline is accounted for: BEIR reports ≈ +0.02 for this kind of
  model over **BM25**; our first stage is already ≈ +0.06–0.11 above BM25, so the general-domain model has
  little left to add and its domain mismatch shows.
- SciFact queries are scientific **claims**, not questions: MS MARCO teaches "does this passage answer the
  question", not "is this the evidence for or against this claim". Negated claims ("… does not cause …")
  lose most (mean Δ nDCG@10 −0.086, 26 queries, small sample), but other claims also lose (−0.023, 274).
- Cost on this CPU: ≈ 1.1 s per query for 20 passages, ≈ 10× the first stage, far above the ≈ 100 ms
  re-ranking budget; serving cost is measured with ONNX in step 6.
- Consequence for the next steps: the case for **in-domain fine-tuning** (claims and nutrition queries,
  hard negatives from our own retriever) is now measured, not assumed.

## Next

- `BAAI/bge-reranker-base` with the same protocol (≈ 6× slower on CPU: GPU notebook or overnight run).
- Fine-tuned cross-encoder (steps 4–5), ONNX serving cost (step 6).
