"""Export first-stage candidates once, then replay any number of re-ranking experiments on them.

runs/<name>/candidates-<split>.jsonl   one line per query:
  {"qid", "query", "candidates": [{"doc_id", "rank", "score", "ranks", "chunk_id", "title", "text"}, ...]}
runs/<name>/candidates-<split>.meta.json   config, index version, retrieval timings

Re-ranking only reorders these lists, so Recall@top_k of a re-ranked run equals the first stage's
(a cheap consistency check), and every re-ranker is compared on exactly the same candidates.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

from cer.config import ExperimentConfig
from cer.data import beir
from cer.eval.runner import build_retriever, percentiles


def candidates_path(cfg: ExperimentConfig, split: str) -> Path:
    return Path(cfg.eval.out_dir) / cfg.name / f"candidates-{split}.jsonl"


def export_candidates(cfg: ExperimentConfig, split: str | None = None, *, progress: bool = True,
                      embedder=None, backend=None, limit: int | None = None) -> dict:
    """Run the first stage on every query of `split` (default: the config's) and write its candidates."""
    split = split or cfg.split
    queries, _ = beir.load(cfg.dataset_dir, split)
    retriever, backend, _ = build_retriever(cfg, progress=progress, embedder=embedder, backend=backend)
    qids = sorted(queries)[:limit]
    out = candidates_path(cfg, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    lat: dict[str, list[float]] = {}
    with open(out, "w", encoding="utf-8") as f:
        for qid in tqdm(qids, disable=not progress, desc=f"{cfg.name}/{split}", unit="q"):
            res = retriever.search(queries[qid])
            for k, v in res.timings_ms.items():
                lat.setdefault(k, []).append(v)
            cands = [{"doc_id": r.doc_id, "rank": i, "score": r.score, "ranks": r.ranks,
                      "chunk_id": r.best_chunk_id, "title": r.best_title, "text": r.best_text}
                     for i, r in enumerate(res.results, start=1)]
            f.write(json.dumps({"qid": qid, "query": queries[qid], "candidates": cands}, ensure_ascii=False) + "\n")
    meta = {"name": cfg.name, "dataset": cfg.dataset, "split": split, "n_queries": len(qids),
            "top_k": cfg.retrieval.top_k, "config": cfg.model_dump(mode="json"),
            "index": {"backend": cfg.backend.type, "alias": cfg.backend.alias,
                      "version": cfg.backend.index_version, "chunks": backend.count()},
            "latency_ms": {k: percentiles(v) for k, v in lat.items()},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"path": str(out), **{k: meta[k] for k in ("name", "split", "n_queries")}}


def read_candidates(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def candidates_to_run(records) -> dict[str, dict[str, float]]:
    """First-stage run (qid -> doc -> score), identical to the run.trec written by `cer eval`."""
    return {r["qid"]: {c["doc_id"]: float(c["score"]) for c in r["candidates"]} for r in records}
