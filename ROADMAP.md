# Roadmap — Clinical Evidence Retrieval

Goal: a retrieval platform for medical evidence built with production constraints in mind
(100M+ chunks, 50+ req/s, p95 < 300 ms): offline indexing, hybrid retrieval, custom re-ranking,
query processing and an evaluation framework that makes every design decision measurable.
Claims about 100M-scale behaviour are **capacity estimates**, measured results come from laptop-scale
corpora and are labelled as such.

Each milestone ends with **measured** results committed to `reports/` and a short write-up. No number is
ever written by hand: every figure comes from a run directory (`runs/<name>/metrics.json`).

| Milestone | Dates (indicative) | Focus | Public release |
|---|---|---|---|
| M1 | 16 → 25 Oct | baselines: BM25 / dense / hybrid on NFCorpus + SciFact | `v0.1` on GitHub |
| M2 | 26 Oct → 3 Nov | re-ranking (off-the-shelf + fine-tuned + ONNX), FR → EN retrieval | `v0.2` |
| M3 | 4 → 10 Nov | guideline documents (HAS), query processing, API with per-stage latency | `v0.3` |
| M4 | 11 → 16 Nov | write-up: design v1.0, capacity model with measured numbers, limitations | `v1.0` |

Rules for every milestone
- One feature = code + tests + config + measured result + one paragraph in `reports/`.
- Never tune on test splits. Use train/dev splits (NFCorpus and SciFact ship `train.tsv`; NFCorpus also `dev.tsv`); report test once.
- Compare systems with `cer compare` (paired permutation test), not only means.
- Keep the in-memory backend working so CI stays green without Elasticsearch.
- Update `docs/DESIGN.md` (and add an ADR) whenever a design decision changes.
- Before each release: `ruff check src tests` and `pytest -q` green, README results table regenerated,
  no data, model weights, PDFs or local paths committed.

---

## M1 — Indexing, hybrid retrieval, evaluation

Code delivered: Elasticsearch backend (int8 HNSW, analyzers, versioned indexes + alias swap), streaming
ingestion, BM25 / dense / hybrid RRF, BEIR loader, metrics (checked against `ranx`), runner, significance
test, CLI, CI.

Remaining work
- [x] Run the full pipeline locally (Windows + Docker Desktop).
- [x] Safety of re-indexing: `create_index` never deletes the index the alias currently serves.
- [x] `cer report` writes the README results table (no manual copy).
- [x] Setup documented in `docs/setup-windows.md`, analysis in `reports/week1.md`.

Exit criteria
- [x] 25 unit/e2e tests + integration tests green locally · [ ] CI green on GitHub
- [x] `reports/results.md`: BM25 / dense (multilingual-e5-base) / hybrid on NFCorpus and SciFact
- [x] hybrid vs BM25 permutation test per dataset (nDCG@10 and Recall@100), written under the table
- [x] ingestion throughput (chunks/s), number of chunks and index size on disk recorded
- [ ] repository public, tag `v0.1`

---

## M2 — Re-ranking and cross-lingual retrieval (priority)

### 2.1 Re-ranking (`src/cer/rerank/`)

Measured constraint (laptop CPU, i7-8565U): re-ranking 50 passages (~220 words, truncated to 256 tokens)
takes ≈ 3.3 s with MiniLM-L6 and ≈ 16 s with bge-reranker-base (512 tokens: 7 s / 27 s). Hence:
re-ranking is **replayed offline on exported candidates** (no Elasticsearch needed, runs on CPU or on a
GPU notebook), quality and latency are reported separately, and fine-tuning runs on a free GPU (Kaggle).

Steps
1. ✅ **Candidate export**: `cer candidates <config> --split train|dev|test` → `runs/<name>/candidates-<split>.jsonl`
   (query, top-100 fused documents with ranks, RRF score and best passage). One retrieval pass, then any
   number of re-ranking experiments (`top_n`, `max_length`, model) on the same candidates.
2. **Reranker interface**: `Reranker.score(query, passages) -> scores`, cross-encoder backend
   (sentence-transformers), `cer rerank <candidates> --model --top-n --max-length` → `runs/<name>/` with
   the same files as `cer eval` (run.trec, metrics.json, per_query.csv) + re-ranking time per query.
   Documents below `top_n` keep their fused order. Unit tests with a fake reranker.
3. **Off-the-shelf baselines**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, ≈ 35 min for both test sets
   at top_n 50) and `BAAI/bge-reranker-base` (GPU notebook, or CPU overnight). `top_n` and `max_length`
   chosen on NFCorpus dev / a SciFact train subset, never on test.
4. **Training data with hard negatives**: `scripts/build_rerank_train.py` from train candidates —
   positives from qrels, negatives = top-50 candidates not judged relevant (ranks 1–3 skipped on NFCorpus,
   where unjudged ≠ non-relevant), 7 train/dev queries whose text equals a test query removed.
   Output under `data/` (derived from BEIR, not committed).
5. **Fine-tuning** (Kaggle GPU): `scripts/train_reranker.py` (sentence-transformers CrossEncoder, BCE,
   1–2 epochs, fixed seed), start from MiniLM-L6 (cheap to serve) then bge-reranker-base if time allows;
   early stopping on NFCorpus dev; model card in `models/<name>/README.md`, weights not committed.
6. **Serving cost**: ONNX export + dynamic int8 quantization (Optimum), latency for 50 passages at batch
   16/32/50 on this CPU and on the notebook GPU, compared with the ≈ 100 ms re-ranking budget (DESIGN §9).
7. **Report**: hybrid vs + MiniLM vs + bge vs + custom (nDCG@10, Recall@100 unchanged by construction,
   p-values vs hybrid), latency per model; `reports/m2-reranking.md`, ADR-004.

Expectations, to read results honestly: in the BEIR paper an MS MARCO cross-encoder on BM25 top-100 gains
only ≈ +0.02 nDCG@10 on NFCorpus and SciFact; NFCorpus Recall@100 is 0.30, so at most a third of the
relevant documents can be moved up. The in-domain fine-tuned model is where a real gain is plausible.

### 2.2 Cross-lingual evaluation (FR → EN)
- `data/fr-queries/<dataset>/queries.jsonl`: French versions of the test queries (machine translation,
  then human review of a sample; method and reviewed share documented in `docs/fr-queries.md`).
- Evaluate BM25, dense and hybrid with FR queries against the EN corpus (BM25 alone is expected to
  collapse: that is the point of the experiment), with and without re-ranking.

Exit criteria
- [ ] re-ranker table (baseline vs custom, ONNX latency); significant gain on ≥ 1 dataset or an honest
      explanation of why not
- [ ] FR → EN table
- [ ] ADR-004 (re-ranker choice and serving)
- [ ] tag `v0.2`

---

## M3 — Medical documents, query processing, serving

### 3.1 Guideline documents (`src/cer/ingest/pdf.py`)
- Parser for French guideline PDFs (PyMuPDF blocks): detect headings (font size/bold), keep the
  section → recommendation structure, extract the grade (A/B/C/AE) when present, never split a
  recommendation across chunks. Keep publication date and status (current / superseded) as metadata.
- 2–3 public HAS recommendation PDFs **downloaded by a script** (PDFs not committed; reuse terms checked).
- Small hand-written FR evaluation set (20–30 questions with the relevant recommendation ids) in
  `data/has-eval/`. Stated clearly as small and illustrative.
- Demo: grade and recency used as filters / boosts.

### 3.2 Query processing (`src/cer/query/`)
- `normalize.py`: Unicode NFKC, accent folding for FR, unit normalisation (mg, mmHg).
- `abbreviations.py`: medical abbreviation dictionary (`resources/abbreviations.yaml`, ~100 entries FR+EN:
  HTA, T2D, MI, BPCO/COPD, HbA1c…), expansion only on standalone tokens, as boosted BM25 `should` clauses.
- Optional: MeSH entry-term expansion (`scripts/build_mesh_synonyms.py`) if time allows.
- Plugged into `Retriever` behind `retrieval.query_processing: none|abbrev|abbrev+mesh`.
- Measure with/without expansion (Recall@100 and nDCG@10: expansion can help recall and hurt precision).

### 3.3 Service (`src/cer/api/`)
- FastAPI `POST /search` {query, filters, top_k, lang} → ranked documents with passage, offsets, source,
  grade and per-stage timings; `GET /health`.
- Models loaded once, ONNX re-ranker, LRU cache for query embeddings, re-ranker timeout with graceful
  degradation (return fused results and flag it).
- Per-stage latency measured on the API (p50/p95), with and without re-ranker. A short load test
  (Locust) at the throughput the laptop sustains; 50 req/s with a CPU cross-encoder is out of laptop
  scope and is covered by the capacity model instead.

Exit criteria
- [ ] guideline ingestion demo with grades as filters, documented
- [ ] query-processing ablation table
- [ ] `reports/latency.md`: per-stage latency with and without re-ranking
- [ ] ADR-005 (query expansion strategy)
- [ ] tag `v0.3`

---

## M4 — Write-up

- `README.md`: one-screen summary (problem, architecture, key results, what changes at 100M).
- `docs/DESIGN.md` v1.0: capacity model **with measured numbers** (bytes/vector, ms/stage), sharding plan
  for 100M chunks, cost of each lever (dims, quantization, `num_candidates`, re-ranker top_n),
  Elasticsearch kNN vs managed vector search (e.g. Vertex AI Vector Search).
- Medical-specific concerns: superseded guidelines, source trust hierarchy, conflicting sources,
  abstention, evaluation with clinician annotations, traceability of citations.
- `docs/LIMITATIONS.md`: benchmarks vs real clinical queries, translation quality, small HAS set, laptop scale.

Exit criteria
- [ ] everything reproducible from the README, one command per experiment
- [ ] CI green, tag `v1.0`

## Later (not planned)
- TREC-COVID + PubMed subset at ~1M chunks (needs a GPU for embeddings), latency vs corpus size curves,
  `num_candidates` sweep, native Elasticsearch `rrf` retriever vs client-side RRF.
- Learned sparse retrieval (SPLADE) and biomedical encoders (MedCPT, BGE-M3) as additional baselines.
- Abstention threshold calibrated on "no-answer" queries.
