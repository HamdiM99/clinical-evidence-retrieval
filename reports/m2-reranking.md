# M2 — Re-ranking

Re-ranking is replayed offline on exported first-stage candidates (`cer candidates`, then `cer rerank`):
every re-ranker sees exactly the same hybrid RRF top 100, scores are cached once and each `top_n` is
derived from the cache. Replaying the test candidates without re-ranking reproduces the M1 hybrid runs
exactly (same rankings, same metrics).

Hardware: laptop CPU (Intel i7-8565U, 4 cores, no GPU). Passages are the best chunk of each document with
its title, ≈ 220 words, truncated with the query to 256 tokens.

> **Update (official SciFact annotations).** The SciFact conclusions below were first read as "the off-the-shelf
> model has a domain mismatch, fine-tuning fixes it". Splitting the test claims by the official SUPPORT /
> CONTRADICT annotation shows that both the MiniLM loss and the C2 gain sit on claims whose BEIR-"relevant"
> document contains no evidence; on claims with real evidence no difference is significant. See
> [Revisiting SciFact with the official annotations](#revisiting-scifact-with-the-official-annotations).

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
| SciFact | −0.028 | 0.055 | −0.025 | 0.178 | −0.025 | 0.153 | 48 / 62 / 190 |

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
| SciFact | MiniLM | **+0.027** | **0.002** | **+0.027** | **0.010** | **+0.028** | **0.004** | 53 / 23 / 224 |

**Decision (rule above): C2 is kept** — not significantly worse than hybrid on either dataset, and
significantly better than the off-the-shelf MiniLM on SciFact.

What the numbers support, and what they do not:
- With BEIR labels, fine-tuning on 608 in-domain claims with a listwise loss beats the MS MARCO model on
  SciFact (+0.027 nDCG@10, p = 0.002) and keeps its NFCorpus behaviour (Δ +0.001). **Revised reading:** this
  gain is +0.054 (p = 0.002) on the 112 claims whose cited documents hold no evidence and +0.012 (p = 0.203)
  on the 188 others; it reflects the benchmark's citation convention more than evidence matching (below).
- No re-ranker beats hybrid RRF significantly on either dataset: on NFCorpus the gain is +0.010
  (p ≈ 0.1), on SciFact C2 is level with the first stage. With 300 queries per test set, effects of
  ≈ 0.01 nDCG@10 are below what this protocol can detect.
- Recall@100 is unchanged (re-ranking the top 20 only): NFCorpus stays capped at 0.30 by the first stage.
- Cost: same architecture as MiniLM (22M parameters), so the same serving cost. Wall-clock re-ranking
  time on this laptop varied between runs (≈ 1.1 s vs ≈ 1.7 s p50 for 20 passages) with machine load;
  serving latency is measured in a controlled benchmark (step 6), not from these replays.

## Serving cost: ONNX export and int8 quantization (step 6)

`cer export-onnx models/C2-listwise-scifact models/C2-listwise-scifact-onnx` writes an ONNX fp32 model and a
dynamically quantized int8 copy (weights int8, activations quantized at run time, no calibration data).

| | size | scores vs PyTorch (30 queries × 20 passages) | validation nDCG@10 NFCorpus dev / SciFact train[:200] |
|---|---|---|---|
| PyTorch fp32 | 86.7 MB | – | 0.3317 / 0.7495 |
| ONNX fp32 | 86.7 MB | max abs diff 1e-5, Kendall τ 1.000, same top 10 on 30/30 | identical |
| ONNX int8 | **21.9 MB** | mean abs diff 0.18, τ 0.937, same top-10 set on 21/30 | **0.3325 / 0.7512** |

Int8 moves the scores but not the mean ranking quality: the validation differences (+0.001, +0.002) are well
inside the noise. The 30-query check above was optimistic about rankings: computed from the cached scores on all
validation queries, int8 keeps the same top-10 set on only 201/324 NFCorpus and 102/200 SciFact queries
(mean Kendall τ 0.94) — same mean, different lists ("silent drift", article v2).

Controlled latency on this laptop (`scripts/bench_reranker.py` → [rerank-latency.md](rerank-latency.md): same
30 queries, warm-up excluded, 4 threads, nothing else running):

| backend | 20 passages p50 / p95 | 50 passages p50 / p95 (best batch) | pairs/s |
|---|---|---|---|
| PyTorch fp32 | 1273 / 2140 ms | 3386 / 4293 ms | ≈ 12–14 |
| ONNX Runtime fp32 | 1482 / 1912 ms | 3365 / 4340 ms | ≈ 11–14 |
| ONNX Runtime int8 | **953 / 1022 ms** | **2165 / 2617 ms** | ≈ 21–22 |

- Exporting to ONNX alone brings nothing on this CPU; **int8 brings ≈ 1.3–1.9× throughput, a much tighter
  p95 and 4× smaller weights, at no measured quality cost.** Batch size barely matters on CPU (compute bound).
- Even in int8, re-ranking 20 passages takes ≈ 1 s on a 15 W laptop CPU: an order of magnitude above the
  ≈ 100 ms re-ranking budget (DESIGN §9). This machine is not a serving target; the budget has to be met on a
  GPU or on server CPUs, measured separately (`scripts/bench_reranker_gpu.py`, Kaggle T4).
- A first benchmark run was discarded: the laptop went to sleep in the middle of it (Windows power log);
  the numbers above come from a clean re-run with sleep prevented.

## GPU latency: does re-ranking fit the budget?

Tesla T4 (Kaggle), full table in [rerank-latency-gpu.md](rerank-latency-gpu.md). Total per query (CPU
tokenization + GPU forward), p50 / p95 in ms, against the ≈ 100 ms re-ranking share of the p95 < 300 ms budget:

| model | precision | 20 passages | 50 passages | 100 passages |
|---|---|---|---|---|
| MiniLM-L6 (= C2) | fp32 | 49 / 61 | 125 / 128 | 263 / 271 |
| MiniLM-L6 (= C2) | **fp16** | **25 / 27** | **60 / 63** | 115 / 125 |
| bge-reranker-base | fp32 | 277 / 280 | 750 / 785 | 1465 / 1576 |
| bge-reranker-base | **fp16** | 76 / 79 | 194 / 197 | 391 / 403 |
| *laptop CPU, C2 int8 (for reference)* | int8 | *953 / 1022* | *2165 / 2617* | – |

- **C2 in fp16 on a T4 fits the budget with margin**: 27 ms p95 at the chosen `top_n = 20`, 63 ms at 50.
  ≈ 38× faster than the laptop CPU in int8.
- **fp16 is the main GPU lever**: 2–4× over fp32 (tensor cores), MiniLM model time 34.7 → 11.4 ms at 20.
- **bge-reranker-base fits only at top 20 in fp16** (79 ms p95), and is 3–6× costlier than MiniLM at equal
  depth: a quality gain would have to be large to justify it.
- **Tokenization becomes the bottleneck**: for MiniLM fp16 it is more than half of the time (14 of 25 ms at
  20 passages, 33 of 60 at 50), on CPU and linear in the number of passages. Lever for serving: tokenize the
  passages once at indexing time and store the token ids with the chunk, so only the query is tokenized online.
- Throughput (estimate, not load-tested): one query at top 20 occupies the GPU ≈ 11 ms, so a single T4 can
  serve ≈ 50 req/s at top 20 if requests are batched and tokenization runs on several CPU workers; to be
  verified with the M3 load test.

## Revisiting SciFact with the official annotations

BEIR marks a SciFact document relevant when the claim cites it. The original release (AllenAI; read with
`scripts/extract_scifact_labels.py`) also says whether each cited abstract SUPPORTS the claim, CONTRADICTS it, or
neither. All numbers below are generated by `docs/article/make_figures.py` (see `docs/article/post-numbers.md`).

- Of the 339 relevant test pairs, 138 SUPPORT, 71 CONTRADICT and **130 (38%) contain no rationale**; for 112 of
  the 300 claims every "relevant" document is of that kind. In the training split, 355 of 919 positives (39%)
  have no rationale either.

| Δ nDCG@10 (BEIR labels) | claims with evidence (188) | claims without evidence (112) |
|---|---|---|
| off-the-shelf MiniLM − hybrid | +0.013 (p = 0.414) | **−0.099 (p < 0.001)** |
| C2 − hybrid | +0.025 (p = 0.125) | −0.045 (p = 0.123) |
| C2 − off-the-shelf MiniLM | +0.012 (p = 0.203) | **+0.054 (p = 0.002)** |

- The off-the-shelf model's SciFact "loss" comes entirely from pushing down cited documents that settle nothing;
  C2's advantage over it is mostly learning to rank them back, a convention present in its training labels.
- Scored on evidence only (relevant = SUPPORT or CONTRADICT, 188 claims): hybrid 0.855, MiniLM 0.867, C2 0.878;
  neither difference with hybrid is significant (C2 +0.022, p = 0.193).
- At rank 1, a document that supports or contradicts the claim appears for 45% (hybrid), 48% (MiniLM) and 49%
  (C2) of claims, with overlapping 95% CIs.
- One study gets the same rank for a claim and its negation (claims 742 / 743, macrolides): the re-rankers score
  relevance, not stance.

Revised conclusion for M2: on these benchmarks a re-ranker after hybrid RRF brings no significant gain, and the
apparent SciFact effects in both directions are artefacts of the relevance labels. The useful outputs of M2 are
the evaluation harness, the risk analysis and the serving measurements. Full write-up:
[docs/article/reranking-little-gain-real-risks.pdf](../docs/article/reranking-little-gain-real-risks.pdf).

## Next

- Train and select re-rankers with stance-aware labels (evidence only) and report evidence-only metrics next to
  BEIR ones.
- `BAAI/bge-reranker-base` with the same protocol (≈ 6× slower on CPU: GPU notebook or overnight run).
