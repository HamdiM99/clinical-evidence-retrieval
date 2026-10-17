"""Hybrid retriever: query embedding -> BM25 + kNN (one round trip) -> chunk->doc max -> RRF.

Every call returns per-stage timings so latency is reported next to quality.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from cer.config import RetrievalConfig
from cer.embed import Embedder
from cer.index import Filters, SearchBackend
from cer.retrieve.fusion import aggregate_max, rrf
from cer.schema import DocResult


@dataclass
class SearchResult:
    results: list[DocResult]
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, backend: SearchBackend, embedder: Embedder | None, cfg: RetrievalConfig):
        if cfg.mode in ("dense", "hybrid") and embedder is None:
            raise ValueError(f"mode={cfg.mode} needs an embedder")
        self.backend, self.embedder, self.cfg = backend, embedder, cfg

    def search(self, query: str, filters: Filters | None = None) -> SearchResult:
        cfg, t = self.cfg, {}
        t0 = time.perf_counter()
        vector = None
        if cfg.mode in ("dense", "hybrid"):
            vector = self.embedder.embed_queries([query])[0]
        t1 = time.perf_counter()
        bm25_hits, knn_hits = self.backend.hybrid_search(
            query, vector,
            k_bm25=cfg.k_bm25 if cfg.mode in ("bm25", "hybrid") else 0,
            k_knn=cfg.k_knn if cfg.mode in ("dense", "hybrid") else 0,
            num_candidates=cfg.num_candidates, filters=filters)
        t2 = time.perf_counter()

        lists = {"bm25": aggregate_max(bm25_hits), "dense": aggregate_max(knn_hits)}
        best = {}
        for name in ("dense", "bm25"):  # bm25 passage wins ties for display
            for did, hit in lists[name]:
                best[did] = hit
        if cfg.mode == "hybrid":
            scores = rrf([[d for d, _ in lists["bm25"]], [d for d, _ in lists["dense"]]],
                         k=cfg.rrf_k, weights=list(cfg.weights))
        else:
            name = cfg.mode
            scores = {d: h.score for d, h in lists[name]}
        ranks = {name: {d: i for i, (d, _) in enumerate(lst, start=1)} for name, lst in lists.items()}
        ordered = sorted(scores, key=lambda d: (-scores[d], d))[: cfg.top_k]
        results = [DocResult(doc_id=d, score=scores[d], best_chunk_id=best[d].chunk_id, best_text=best[d].text,
                             best_title=best[d].title,
                             ranks={n: ranks[n][d] for n in ranks if d in ranks[n]}) for d in ordered]
        t3 = time.perf_counter()
        t["embed"], t["search"], t["fusion"], t["total"] = (
            (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3)
        return SearchResult(results, t)
