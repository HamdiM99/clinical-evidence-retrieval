# ADR-002 · Elasticsearch as the single retrieval engine

**Status:** accepted (week 1)

## Context
The target system needs BM25 with language analyzers, ANN vector search, metadata filters, versioned
indexes and operations at 100M+ documents.

## Decision
Elasticsearch 8.x hosts both the lexical and the vector index for the same chunk documents.

## Rationale
- One engine, one document, one filter language: no cross-store consistency problem.
- HNSW with int8 quantization (`int8_hnsw`) reduces vector memory ≈ 4×.
- Aliases give atomic, zero-downtime index swaps and rollbacks.
- Mature operations story (sharding, replicas, snapshots).

## Alternatives
- **OpenSearch**: equivalent model; the backend interface allows a port.
- **Solr**: strong lexical engine; dense search exists but is less central to its ecosystem.
- **Vertex AI Vector Search** (managed ANN): good at scale, but lexical search and filters live elsewhere.
- **Dedicated vector DB + separate BM25**: two systems to keep consistent; justified only if ANN
  performance in ES becomes the bottleneck.

## Consequences
- Vector memory sizing drives the search-tier hardware (see DESIGN §9).
- An in-memory backend with the same interface is kept for tests and CI without Elasticsearch.
