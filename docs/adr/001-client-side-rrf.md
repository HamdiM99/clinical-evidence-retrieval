# ADR-001 · Client-side RRF over a single `_msearch`

**Status:** accepted (week 1)

## Context
Hybrid retrieval needs a lexical list and a vector list fused into one ranking. Options:
1. Elasticsearch native `rrf` retriever (server-side).
2. Linear combination of scores inside one query (`bool` + `knn` with boosts).
3. Two searches in one `_msearch` request, fused in the application with RRF.

## Decision
Option 3 by default; option 1 kept as a benchmarked alternative.

## Rationale
- BM25 scores are unbounded and query-dependent, cosine scores are bounded: linear combination needs
  per-query-family calibration (option 2). RRF only uses ranks.
- Chunk → document aggregation (max) must happen **before** fusion; server-side RRF fuses chunks.
- Fusion logic is unit-testable and identical across backends (in-memory, ES, OpenSearch).
- One `_msearch` keeps a single network round trip; ES runs both searches in parallel.
- Availability of server-side RRF depends on version and licence tier; client-side has no such constraint.

## Consequences
- Slightly more data transferred (two candidate lists). Bounded by `k1`, `k2` (default 100).
- The native retriever is evaluated in week 3 for latency comparison.
