"""Run one experiment config on a BEIR dataset and write reproducible artifacts.

runs/<name>/
  config.yaml      the exact config used
  run.trec         TREC run file (qid Q0 docid rank score tag)
  metrics.json     mean metrics + latency percentiles + index info
  per_query.csv    one row per query: metrics, latency, retriever overlap
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from cer.config import ExperimentConfig
from cer.data import beir
from cer.embed import build_embedder
from cer.eval.metrics import per_query
from cer.index import build_backend
from cer.ingest.pipeline import ingest
from cer.retrieve.retriever import Retriever


def write_trec(run: dict[str, dict[str, float]], path: Path, tag: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(run):
            f.writelines(f"{qid} Q0 {did} {rank} {score:.6f} {tag}\n" for rank, (did, score) in enumerate(sorted(run[qid].items(), key=lambda x: (-x[1], x[0])), start=1))


def read_trec(path: str | Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, did, _, score, _ = line.split()
            run.setdefault(qid, {})[did] = float(score)
    return run


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean())}


def run_experiment(cfg: ExperimentConfig, *, progress: bool = True, embedder=None, backend=None) -> dict:
    ds = cfg.dataset_dir
    queries, qrels = beir.load(ds, cfg.split)
    needs_embedder = cfg.retrieval.mode in ("dense", "hybrid")
    embedder = embedder if embedder is not None else (build_embedder(cfg.embedder) if needs_embedder else None)
    backend = backend if backend is not None else build_backend(cfg.backend)

    ingest_info = None
    if cfg.backend.type == "memory":  # in-process index: build it now
        rep = ingest(beir.iter_corpus(ds), backend, embedder, version=cfg.backend.index_version,
                     max_words=cfg.chunking.max_words, overlap_words=cfg.chunking.overlap_words,
                     batch_size=cfg.embedder.batch_size, progress=progress)
        ingest_info = rep.__dict__
    retriever = Retriever(backend, embedder, cfg.retrieval)

    run: dict[str, dict[str, float]] = {}
    lat: dict[str, list[float]] = {}
    overlap: dict[str, dict[str, int]] = {}
    for qid, text in tqdm(sorted(queries.items()), disable=not progress, desc=cfg.name, unit="q"):
        res = retriever.search(text)
        run[qid] = {r.doc_id: r.score for r in res.results}
        for k, v in res.timings_ms.items():
            lat.setdefault(k, []).append(v)
        overlap[qid] = {"both": sum(1 for r in res.results[:10] if len(r.ranks) == 2),
                        "bm25_only": sum(1 for r in res.results[:10] if list(r.ranks) == ["bm25"]),
                        "dense_only": sum(1 for r in res.results[:10] if list(r.ranks) == ["dense"])}

    pq = per_query(run, qrels, cfg.eval.metrics)
    means = {m: (sum(v.values()) / len(v) if v else 0.0) for m, v in pq.items()}
    out = Path(cfg.eval.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    write_trec(run, out / "run.trec", cfg.name)
    summary = {
        "name": cfg.name, "dataset": cfg.dataset, "split": cfg.split, "n_queries": len(next(iter(pq.values()), {})),
        "metrics": means, "latency_ms": {k: percentiles(v) for k, v in lat.items()},
        "index": {"backend": cfg.backend.type, "alias": cfg.backend.alias, "version": cfg.backend.index_version,
                  "chunks": backend.count()},
        "ingest": ingest_info, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "query", *cfg.eval.metrics, "total_ms", "top10_both", "top10_bm25_only", "top10_dense_only"])
        for i, qid in enumerate(sorted(pq[cfg.eval.metrics[0]])):
            idx = sorted(queries).index(qid)
            w.writerow([qid, queries[qid], *[f"{pq[m][qid]:.4f}" for m in cfg.eval.metrics],
                        f"{lat['total'][idx]:.2f}", *overlap[qid].values()])
    return summary
