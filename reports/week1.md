# M1 — Baselines: BM25, dense, hybrid RRF

Setup: BEIR test splits (NFCorpus 323 queries, SciFact 300 queries), Elasticsearch 8.15.3 single node
(Docker, WSL 2), `intfloat/multilingual-e5-base` on CPU (Intel i7-8565U, 4 cores, no GPU), chunks of
250 words with 40 words overlap, BM25 top-100 + kNN top-100 (`num_candidates` 200, int8 HNSW),
RRF k = 60, chunk → document max before fusion. Configs: `configs/{nfcorpus,scifact}_{bm25,dense,hybrid}.yaml`.
The full table is in [results.md](results.md).

## Results

| dataset | bm25 nDCG@10 | dense nDCG@10 | hybrid nDCG@10 | hybrid Recall@100 |
|---|---|---|---|---|
| nfcorpus | 0.296 | 0.327 | **0.353** | 0.301 |
| scifact | 0.619 | 0.704 | **0.730** | 0.948 |

Paired permutation test (10,000 permutations; p < 0.001 means the smallest value reachable):

| dataset | comparison | Δ nDCG@10 | p | Δ Recall@100 | p |
|---|---|---|---|---|---|
| nfcorpus | hybrid vs bm25 | +0.057 | < 0.001 | +0.066 | < 0.001 |
| nfcorpus | hybrid vs dense | +0.026 | < 0.001 | +0.022 | < 0.001 |
| scifact | hybrid vs bm25 | +0.111 | < 0.001 | +0.044 | < 0.001 |
| scifact | hybrid vs dense | +0.026 | 0.050 | +0.018 | 0.105 |

## Analysis

- Hybrid RRF is the best system on both datasets and every metric. It is clearly better than BM25
  (per query: 148 wins / 45 losses on NFCorpus, 100 / 19 on SciFact); against dense alone the gain is
  significant on NFCorpus, borderline on SciFact for nDCG@10 (p = 0.05) and not significant for Recall@100.
- The two retrievers are complementary: their top-10 lists overlap little (mean Jaccard 0.19 on both
  datasets). Among relevant documents in a top 10, NFCorpus has 423 found by both, 255 by BM25 only and
  339 by dense only; on SciFact, 231 / 15 / 44 — dense already covers most of what BM25 finds there.
- `top10_bm25_only` / `top10_dense_only` in `per_query.csv` count hybrid top-10 documents present in only
  one **candidate list** (top 100). They are low (NFCorpus 0.4 / 1.9, SciFact 0.2 / 0.1 per query) because
  RRF favours documents that both retrievers return; the top-10 overlap above is the more telling number.
- NFCorpus stays hard (short queries, graded judgments): 88 queries out of 323 have nDCG@10 = 0 with
  hybrid (108 with BM25). Recall@100 of 0.30 means a re-ranker on the top 100 can only reorder a third of
  the relevant documents: candidate generation, not only re-ranking, needs work there.
- BM25 returns no document at all for 15 NFCorpus queries: single words absent from the corpus vocabulary,
  among them drug brand names (`Fosamax` → alendronate, `Mevacor` → lovastatin) and foods (`eggnog`,
  `halibut`). They score 0 (metrics are averaged over all 323 queries), while dense retrieval still returns
  candidates. This is a direct case for brand → generic (INN) and synonym expansion in query processing (M3).
- Our BM25 is below the published BEIR BM25 numbers (Anserini: ≈ 0.325 NFCorpus, ≈ 0.665 SciFact).
  Likely causes: Elasticsearch `english` analyzer and a `best_fields` multi-match over title and text,
  vs Anserini's settings. Not tuned here on purpose (baselines; tuning belongs on train/dev splits).

## Indexing

| dataset | documents | chunks | ingest time | throughput | embedding share | index size |
|---|---|---|---|---|---|---|
| nfcorpus | 3,633 | 4,811 | 70 min | 1.1 chunks/s | 99.4 % | 84 MB |
| scifact | 5,183 | 6,428 | 86 min | 1.2 chunks/s | 99.5 % | 113 MB |

Embedding on a laptop CPU dominates indexing (bulk indexing itself takes ≈ 20 s per dataset). At this
rate 100M chunks would need ≈ 2.7 years on this machine: production indexing must run on GPUs and be
incremental.

## Latency (laptop, not a production measurement)

p50 per query: BM25 ≈ 62–76 ms; dense and hybrid ≈ 127–150 ms, of which ≈ 50–72 ms is query embedding on
CPU and ≈ 72–79 ms the `_msearch` round trip; fusion < 1 ms. On corpora of a few thousand chunks the search
time is mostly fixed overhead (Docker/WSL networking, Python client), so it says nothing yet about
behaviour at scale; per-stage latency on a larger index is measured in M3.

## Limitations

- Two small English benchmarks; no French queries yet (M2) and no clinical guidelines yet (M3).
- Single run per configuration (retrieval is deterministic; latency varies with machine load).
- The top-10 overlap statistics on NFCorpus use the 308 queries for which BM25 returns results.
