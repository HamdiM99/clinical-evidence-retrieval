"""Elasticsearch 8.x backend.

- Versioned physical indexes `{alias}-v{N}` behind a read alias `{alias}` (zero-downtime swap, rollback).
- Bulk load with refresh disabled and 0 replicas, restored in `finalize`.
- BM25 and kNN sent in ONE `_msearch` round trip, fused client side (ADR-001).

Query/mapping builders are pure functions so they are unit-tested without a cluster.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cer.config import BackendConfig
from cer.index import Filters
from cer.schema import Chunk, Hit

SOURCE_FIELDS = ["chunk_id", "doc_id", "title", "text", "source", "lang", "published", "evidence_grade",
                 "start_char", "end_char", "position"]


# ------------------------------------------------------------------ pure builders
def build_index_body(dim: int, shards: int = 1, m: int = 16, ef_construction: int = 100) -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": shards,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
            "analysis": {"analyzer": {"default": {"type": "english"}}},
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "position": {"type": "integer"},
                "section": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "english",
                          "fields": {"fr": {"type": "text", "analyzer": "french"}}},
                "text": {"type": "text", "analyzer": "english",
                         "fields": {"fr": {"type": "text", "analyzer": "french"},
                                    "raw": {"type": "text", "analyzer": "standard"}}},
                "source": {"type": "keyword"},
                "lang": {"type": "keyword"},
                "published": {"type": "date", "ignore_malformed": True},
                "evidence_grade": {"type": "keyword"},
                "start_char": {"type": "integer", "index": False},
                "end_char": {"type": "integer", "index": False},
                "embedding": {
                    "type": "dense_vector", "dims": dim, "index": True, "similarity": "cosine",
                    "index_options": {"type": "int8_hnsw", "m": m, "ef_construction": ef_construction},
                },
            },
        },
    }


def build_filter_clauses(filters: Filters | None) -> list[dict[str, Any]]:
    clauses = []
    for field, value in (filters or {}).items():
        clauses.append({"terms": {field: value}} if isinstance(value, list) else {"term": {field: value}})
    return clauses


def build_bm25_query(text: str, size: int, fields: list[str], filters: Filters | None = None) -> dict[str, Any]:
    return {
        "size": size,
        "_source": SOURCE_FIELDS,
        "query": {"bool": {
            "must": [{"multi_match": {"query": text, "fields": fields, "type": "best_fields"}}],
            "filter": build_filter_clauses(filters),
        }},
    }


def build_knn_query(vector: Sequence[float], k: int, num_candidates: int,
                    filters: Filters | None = None) -> dict[str, Any]:
    knn: dict[str, Any] = {"field": "embedding", "query_vector": [float(x) for x in vector],
                           "k": k, "num_candidates": max(num_candidates, k)}
    clauses = build_filter_clauses(filters)
    if clauses:
        knn["filter"] = clauses  # pre-filtering inside HNSW search, not post-filtering
    return {"size": k, "_source": SOURCE_FIELDS, "knn": knn}


def build_msearch(index: str, bm25: dict | None, knn: dict | None) -> list[dict[str, Any]]:
    body: list[dict[str, Any]] = []
    for q in (bm25, knn):
        if q is not None:
            body += [{"index": index}, q]
    return body


def parse_hits(resp: dict[str, Any]) -> list[Hit]:
    out = []
    for h in resp.get("hits", {}).get("hits", []):
        s = h.get("_source", {})
        out.append(Hit(chunk_id=s.get("chunk_id", h.get("_id")), doc_id=s["doc_id"], score=float(h["_score"]),
                       text=s.get("text", ""), title=s.get("title", "")))
    return out


# ------------------------------------------------------------------ backend
class ElasticsearchBackend:
    def __init__(self, cfg: BackendConfig, client=None):
        from elasticsearch import Elasticsearch

        self.cfg = cfg
        self.es = client or Elasticsearch(cfg.url, request_timeout=cfg.request_timeout)

    def index_name(self, version: int) -> str:
        return f"{self.cfg.alias}-v{version}"

    def serving_indexes(self) -> list[str]:
        alias = self.cfg.alias
        return list(self.es.indices.get_alias(name=alias)) if self.es.indices.exists_alias(name=alias) else []

    def create_index(self, dim: int, version: int) -> str:
        name = self.index_name(version)
        if self.es.indices.exists(index=name):
            if name in self.serving_indexes():  # never drop the index the alias serves
                raise ValueError(f"{name} is served by alias {self.cfg.alias}; "
                                 f"bump backend.index_version to rebuild")
            self.es.indices.delete(index=name)  # leftover of an interrupted build
        body = build_index_body(dim)
        self.es.indices.create(index=name, settings=body["settings"], mappings=body["mappings"])
        return name

    def index_chunks(self, chunks: Sequence[Chunk], vectors: np.ndarray, index_name: str) -> int:
        from elasticsearch import helpers

        def actions():
            for c, v in zip(chunks, vectors):
                doc = c.to_index_doc()
                doc["embedding"] = [float(x) for x in v]
                yield {"_op_type": "index", "_index": index_name, "_id": c.chunk_id, "_source": doc}

        ok = 0
        for success, info in helpers.streaming_bulk(self.es, actions(), chunk_size=500, raise_on_error=False):
            if success:
                ok += 1
            else:
                raise RuntimeError(f"bulk indexing failed: {info}")
        return ok

    def finalize(self, index_name: str) -> None:
        self.es.indices.put_settings(index=index_name,
                                     settings={"refresh_interval": "1s", "number_of_replicas": 0})
        self.es.indices.refresh(index=index_name)
        self.es.indices.forcemerge(index=index_name, max_num_segments=1, wait_for_completion=True)

    def swap_alias(self, index_name: str) -> None:
        alias = self.cfg.alias
        actions: list[dict[str, Any]] = [{"remove": {"index": old, "alias": alias}}
                                         for old in self.serving_indexes()]
        actions.append({"add": {"index": index_name, "alias": alias}})
        self.es.indices.update_aliases(actions=actions)  # atomic

    def count(self) -> int:
        return int(self.es.count(index=self.cfg.alias)["count"])

    def hybrid_search(self, text, vector, k_bm25, k_knn, num_candidates, filters=None):
        bm25 = build_bm25_query(text, k_bm25, self.cfg.bm25_fields, filters) if k_bm25 > 0 else None
        knn = build_knn_query(vector, k_knn, num_candidates, filters) if (k_knn > 0 and vector is not None) else None
        resp = self.es.msearch(searches=build_msearch(self.cfg.alias, bm25, knn))["responses"]
        it = iter(resp)
        bm25_hits = parse_hits(next(it)) if bm25 is not None else []
        knn_hits = parse_hits(next(it)) if knn is not None else []
        for r in resp:
            if "error" in r:
                raise RuntimeError(f"search failed: {r['error']}")
        return bm25_hits, knn_hits
