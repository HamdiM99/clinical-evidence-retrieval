# Re-ranker latency (GPU)

Measured with `scripts/bench_reranker_gpu.py` on a Kaggle notebook, NVIDIA Tesla T4 (16 GB), one GPU.
Table as printed by the script (`gpu_latency.json` in the notebook output). Passages: real first-stage
candidates of 33 NFCorpus dev queries (`bench_passages.jsonl`), 30 measured after 3 warm-up queries, max
length 256 tokens. Per query: tokenization on CPU + forward pass on GPU (CUDA-synchronised). The fine-tuned
C2 has the MiniLM-L6 architecture, hence the same latency as `ms-marco-MiniLM-L-6-v2`.

| model | precision | passages | batch | tokenize p50 | model p50 | total p50 | total p95 |
|---|---|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | fp32 | 20 | 20 | 14.2 | 34.7 | 48.8 | 60.5 |
| ms-marco-MiniLM-L-6-v2 | fp32 | 50 | 50 | 33.0 | 91.7 | 125.1 | 128.2 |
| ms-marco-MiniLM-L-6-v2 | fp32 | 100 | 100 | 64.2 | 197.7 | 262.8 | 271.2 |
| ms-marco-MiniLM-L-6-v2 | fp32 | 100 | 32 | 70.1 | 201.6 | 273.5 | 285.8 |
| ms-marco-MiniLM-L-6-v2 | fp16 | 20 | 20 | 13.9 | 11.4 | 25.4 | 26.6 |
| ms-marco-MiniLM-L-6-v2 | fp16 | 50 | 50 | 33.4 | 26.5 | 59.7 | 63.1 |
| ms-marco-MiniLM-L-6-v2 | fp16 | 100 | 100 | 63.7 | 52.4 | 114.5 | 125.0 |
| ms-marco-MiniLM-L-6-v2 | fp16 | 100 | 32 | 67.9 | 57.8 | 124.8 | 133.4 |
| bge-reranker-base | fp32 | 20 | 20 | 13.5 | 262.7 | 276.6 | 279.5 |
| bge-reranker-base | fp32 | 50 | 50 | 29.8 | 720.6 | 750.3 | 785.1 |
| bge-reranker-base | fp32 | 100 | 100 | 55.5 | 1408.1 | 1465.4 | 1576.1 |
| bge-reranker-base | fp32 | 100 | 32 | 60.9 | 1485.0 | 1551.1 | 1607.3 |
| bge-reranker-base | fp16 | 20 | 20 | 12.5 | 63.2 | 75.8 | 78.8 |
| bge-reranker-base | fp16 | 50 | 50 | 29.6 | 163.7 | 193.7 | 196.5 |
| bge-reranker-base | fp16 | 100 | 100 | 55.9 | 336.6 | 391.0 | 403.2 |
| bge-reranker-base | fp16 | 100 | 32 | 60.8 | 348.3 | 406.8 | 420.1 |

All times in ms.
