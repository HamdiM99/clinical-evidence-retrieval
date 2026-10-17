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

## Training data for the custom re-ranker

Built by `scripts/build_rerank_train.py` from the train candidates of the hybrid first stage
(`data/rerank-train/`, derived from BEIR, not committed; counts in `stats.json`).

| | NFCorpus train | SciFact train (minus 200 held out) |
|---|---|---|
| queries used | 2,582 | 608 |
| removed: text identical to a test or validation query | 8 | 1 |
| positives (max 8 per query) | 17,081, of which 9,798 retrieved in the top 50 | 687, of which 641 retrieved |
| hard negatives (top 50, not judged relevant) | 66,523, ranks 1–3 skipped | 2,748 |
| pairs | 83,604 | 3,435 × 3 (repeated to balance) |

- Train qrels have a single grade (1) on both datasets: binary labels.
- Negatives are the highest-ranked non-relevant candidates (hardest first), 4 per positive. On NFCorpus the
  top 3 are skipped because judgments are incomplete (≈ 40 relevant documents per query, derived from
  links, so a top-ranked unjudged document is often relevant); SciFact links each claim to its source.
- Positives not retrieved by the first stage use their document's first chunk: the model also learns
  from relevant documents the retriever misses (43 % of NFCorpus positives).
- Validation = the tuning sets of step 3 (NFCorpus dev 324 queries, SciFact train[:200]), with their
  first-stage top 20 and qrels; checked: no train query (id or text) in test or validation.

## Fine-tuning, run 1: pointwise BCE (no gain)

`scripts/train_reranker.py`, base `cross-encoder/ms-marco-MiniLM-L-6-v2`, both datasets, pointwise BCE,
lr 2e-5 (10 % warm-up, linear decay), 32 pairs per step, 1 epoch (2,935 steps), Kaggle Tesla T4, ≈ 6 min.
Validation nDCG@10 at top 20 (step 0 = base model; it equals the step-3 tuning numbers to 4 decimals,
so the notebook evaluation and the local replay agree):

| step | NFCorpus dev | SciFact train[:200] | mean |
|---|---|---|---|
| first stage (hybrid) | 0.3222 | 0.7497 | 0.5359 |
| 0 (base MiniLM) | 0.3306 | 0.7302 | **0.5304** |
| 500 | 0.2990 | 0.6622 | 0.4806 |
| 1000 | 0.3031 | 0.5528 | 0.4279 |
| 1500 | 0.3244 | 0.6711 | 0.4977 |
| 2000 | 0.3258 | 0.6817 | 0.5037 |
| 2500 | 0.3335 | 0.6913 | 0.5124 |
| 2935 | 0.3271 | 0.6635 | 0.4953 |

No checkpoint beats the base model: the base model is kept. NFCorpus forgets then recovers to the base
level (+0.003 at best); SciFact degrades throughout (down to 0.553). Hypotheses, tested next one at a time:
(A) learning rate too high for an already good model (early drop); (B) the pointwise objective — absolute
0/1 targets on noisy labels — instead of ranking the documents of one query against each other;
(C) conflicting notions of relevance: 89 % of the pairs are NFCorpus (broad topical relevance), while
SciFact needs the evidence for one claim.

## Fine-tuning, ablation: what makes it work

Same base model, data, validation and selection rule; one change per run (Kaggle T4, `docs/kaggle-training.md`).
Validation nDCG@10 of the best checkpoint (step 0 = base model, kept when never beaten):

| run | change | NFCorpus dev | SciFact train[:200] | mean | best step |
|---|---|---|---|---|---|
| first stage (hybrid, no re-ranking) | – | 0.3222 | 0.7497 | 0.5359 | – |
| base MiniLM | – | 0.3306 | 0.7302 | 0.5304 | – |
| 1 | BCE, lr 2e-5, both datasets | = base | = base | = base | 0 / 2935 |
| A | BCE, **lr 5e-6** | = base | = base | = base | 0 / 2935 |
| B | **listwise** (1 pos + 7 hard neg, softmax), lr 1e-5, both | 0.3245 | 0.7434 | 0.5340 | 2500 / 4786 |
| C1 | listwise, **NFCorpus only** | = base | = base | = base | 0 / 4271 |
| C2 | listwise, **SciFact only** (608 claims) | **0.3317** | **0.7495** | **0.5406** | 400 / 516 |

- **NFCorpus training data is what fails**, not the learning rate or the loss: trained alone on NFCorpus
  (C1), the model never improves on NFCorpus; mixed in (runs 1, A, B) it degrades SciFact. NFCorpus train
  labels are broad (≈ 40 "relevant" documents per query, derived from links): they do not teach a
  cross-encoder already trained on MS MARCO anything it does not know, and pull it away from precise
  evidence matching. A lower learning rate (A) only delays the SciFact collapse (from step ≈ 750).
- **The listwise objective helps** (B > A with the same data) but does not compensate for the labels.
- **C2 is the only clear gain**: 608 SciFact claims, 3 minutes on a T4. It removes the SciFact regression of
  the off-the-shelf model (0.730 → 0.750) without forgetting NFCorpus (0.331 → 0.332).
- Caveat: on SciFact, C2 reaches the level of the first stage **without** re-ranking (0.750), so its gain
  over "no re-ranking" comes from NFCorpus (as for the base model). C2 was selected among 10 evaluations of
  the same 200 queries: the validation gain is optimistic, test decides.

Decision rule for test, fixed before running it: C2 is kept as the custom re-ranker if it is not
significantly worse than hybrid RRF on either dataset **and** beats the base MiniLM on SciFact.

## Custom re-ranker C2 on test (reported once)

Local replay of the downloaded checkpoint first reproduces the Kaggle validation exactly (NFCorpus dev
0.3317, SciFact train[:200] 0.7495). Then test, top_n = 20:

| dataset | hybrid RRF | + MiniLM (MS MARCO) | + C2 (fine-tuned on SciFact) |
|---|---|---|---|
| NFCorpus nDCG@10 | 0.3528 | 0.3615 | **0.3629** |
| NFCorpus MRR@10 | 0.5596 | 0.5877 | 0.5861 |
| SciFact nDCG@10 | **0.7298** | 0.7013 | 0.7288 |
| SciFact MRR@10 | 0.6970 | 0.6721 | **0.6988** |

Paired permutation tests (10,000 permutations), C2 minus reference:

| dataset | vs | Δ nDCG@10 | p | Δ MRR@10 | p | Δ MAP@100 | p | win / loss / tie |
|---|---|---|---|---|---|---|---|---|
| NFCorpus | hybrid | +0.010 | 0.098 | +0.027 | 0.082 | +0.006 | 0.123 | 106 / 86 / 131 |
| NFCorpus | MiniLM | +0.001 | 0.607 | −0.002 | 0.804 | −0.001 | 0.797 | 72 / 62 / 189 |
| SciFact | hybrid | −0.001 | 0.952 | +0.002 | 0.928 | +0.003 | 0.893 | 62 / 52 / 186 |
| SciFact | MiniLM | **+0.028** | **0.002** | **+0.027** | **0.010** | **+0.028** | **0.004** | 53 / 23 / 224 |

**Decision (rule above): C2 is kept** — not significantly worse than hybrid on either dataset, and
significantly better than the off-the-shelf MiniLM on SciFact.

What the numbers support, and what they do not:
- Fine-tuning on 608 in-domain claims with a listwise loss **fixes the domain mismatch** of the MS MARCO
  model on SciFact (+0.028 nDCG@10, p = 0.002) and keeps its NFCorpus behaviour (Δ +0.001).
- No re-ranker beats hybrid RRF significantly on either dataset: on NFCorpus the gain is +0.010
  (p ≈ 0.1), on SciFact C2 is level with the first stage. With 300 queries per test set, effects of
  ≈ 0.01 nDCG@10 are below what this protocol can detect.
- Recall@100 is unchanged (re-ranking the top 20 only): NFCorpus stays capped at 0.30 by the first stage.
- Cost: same architecture as MiniLM (22M parameters), so the same serving cost. Wall-clock re-ranking
  time on this laptop varied between runs (≈ 1.1 s vs ≈ 1.7 s p50 for 20 passages) with machine load;
  serving latency is measured in a controlled benchmark (step 6), not from these replays.

## Next

- `BAAI/bge-reranker-base` with the same protocol (≈ 6× slower on CPU: GPU notebook or overnight run).
- ONNX export + int8 quantization of C2 and a controlled latency benchmark (step 6).
