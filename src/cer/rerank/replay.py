"""Replay re-ranking on exported candidates.

1. `score_candidates` runs the reranker on the first `depth` candidates of every query and caches the
   scores (runs/<first stage>/rerank-cache/<model>-L<max_length>-<split>.jsonl). The cache is plain
   JSONL, so it can also be produced on a GPU notebook and copied back.
2. `rerank_run` turns cached scores into a run for any top_n <= depth: the first top_n documents are
   sorted by reranker score (ties: fused rank), the others keep their fused order below them.
   It writes the same artifacts as `cer eval`, so `cer report` and `cer compare` work unchanged.

Latency: the reranker time is measured per query at `depth`; for a smaller top_n it is scaled linearly
(cost is proportional to the number of pairs) and flagged as estimated in metrics.json.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from cer.config import RerankConfig
from cer.data import beir
from cer.eval.metrics import per_query
from cer.eval.runner import percentiles, write_trec
from cer.rerank import Reranker, passage_text
from cer.rerank.candidates import read_candidates


def model_slug(model: str) -> str:
    return model.rstrip("/").split("/")[-1].lower()


def cache_path(candidates: Path, cfg: RerankConfig) -> Path:
    split = candidates.stem.removeprefix("candidates-")
    return candidates.parent / "rerank-cache" / f"{model_slug(cfg.model)}-L{cfg.max_length}-{split}.jsonl"


def read_cache(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["qid"]: r for r in map(json.loads, f) if r}


def score_candidates(candidates: Path, reranker: Reranker, cfg: RerankConfig, depth: int, *,
                     limit: int | None = None, progress: bool = True) -> Path:
    """Score the first `depth` candidates of each query once; reuse the cache when it is deep enough."""
    out = cache_path(candidates, cfg)
    if out.exists():
        cached = read_cache(out)
        n_records = sum(1 for _ in read_candidates(candidates)) if limit is None else limit
        if len(cached) >= n_records and all(r["depth"] >= depth for r in cached.values()):
            return out
    out.parent.mkdir(parents=True, exist_ok=True)
    records = list(read_candidates(candidates))[:limit]
    if records and records[0]["candidates"]:  # untimed warm-up: model loading and first-call overhead
        c0 = records[0]["candidates"][0]
        reranker.score(records[0]["query"], [passage_text(c0["title"], c0["text"])])
    with open(out, "w", encoding="utf-8") as f:
        for r in tqdm(records, disable=not progress, desc=f"rerank {model_slug(cfg.model)}", unit="q"):
            cands = r["candidates"][:depth]
            t0 = time.perf_counter()
            scores = reranker.score(r["query"], [passage_text(c["title"], c["text"]) for c in cands])
            ms = (time.perf_counter() - t0) * 1e3
            f.write(json.dumps({"qid": r["qid"], "depth": depth, "doc_ids": [c["doc_id"] for c in cands],
                                "scores": scores, "ms": ms}) + "\n")
    return out


def rerank_order(fused: list[str], scored: dict[str, float], top_n: int) -> list[str]:
    pos = {d: i for i, d in enumerate(fused)}
    head = sorted(fused[:top_n], key=lambda d: (-scored[d], pos[d]))
    return head + fused[top_n:]


def rerank_run(candidates: Path, cfg: RerankConfig, *, out_dir: Path = Path("runs"), name: str | None = None,
               metrics: list[str] | None = None, limit: int | None = None) -> dict:
    meta = json.loads(candidates.with_suffix(".meta.json").read_text(encoding="utf-8"))
    split, dataset = meta["split"], meta["dataset"]
    data_dir = Path(meta["config"]["data_dir"]) / dataset
    metrics = metrics or meta["config"]["eval"]["metrics"]
    cache = read_cache(cache_path(candidates, cfg))
    records = list(read_candidates(candidates))[:limit]

    run: dict[str, dict[str, float]] = {}
    lat: dict[str, list[float]] = {"rerank": [], "first_stage": [], "total": []}
    changed: dict[str, int] = {}
    rerank_ms_by_q: dict[str, float] = {}
    estimated = False
    for r in records:
        c = cache[r["qid"]]
        if c["depth"] < min(cfg.top_n, len(r["candidates"])):
            raise ValueError(f"cache depth {c['depth']} < top_n {cfg.top_n}: rescore with a larger depth")
        fused = [x["doc_id"] for x in r["candidates"]]
        order = rerank_order(fused, dict(zip(c["doc_ids"], c["scores"])), cfg.top_n)
        run[r["qid"]] = {d: float(len(order) - i) for i, d in enumerate(order)}  # rank-preserving scores
        changed[r["qid"]] = len(set(order[:10]) - set(fused[:10]))
        n_pairs = min(cfg.top_n, len(c["doc_ids"]))
        rerank_ms = c["ms"] * n_pairs / max(len(c["doc_ids"]), 1)
        estimated |= n_pairs != len(c["doc_ids"])
        lat["rerank"].append(rerank_ms)
        rerank_ms_by_q[r["qid"]] = rerank_ms
        if "timings_ms" in r:
            lat["first_stage"].append(r["timings_ms"]["total"])
            lat["total"].append(r["timings_ms"]["total"] + rerank_ms)

    qrels = beir.load_qrels(data_dir, split)
    pq = per_query(run, {q: qrels.get(q, {}) for q in run}, metrics)
    name = name or f"{meta['name']}-{model_slug(cfg.model)}-L{cfg.max_length}-top{cfg.top_n}" + (
        "" if split == "test" else f"-{split}")
    out = out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({"rerank": cfg.model_dump(mode="json"),
                                                     "candidates": str(candidates),
                                                     "first_stage": meta["config"]}, sort_keys=False),
                                     encoding="utf-8")
    write_trec(run, out / "run.trec", name)
    summary = {
        "name": name, "dataset": dataset, "split": split, "n_queries": len(run),
        "metrics": {m: (sum(v.values()) / len(v) if v else 0.0) for m, v in pq.items()},
        "latency_ms": {k: percentiles(v) for k, v in lat.items() if v},
        "rerank_latency": "estimated (scaled from the scored depth)" if estimated else "measured",
        "rerank": cfg.model_dump(mode="json"), "first_stage": meta["name"], "index": meta["index"],
        "candidates": str(candidates), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "query", *metrics, "rerank_ms", "top10_new_vs_fused"])
        queries = {r["qid"]: r["query"] for r in records}
        for qid in sorted(run):
            w.writerow([qid, queries[qid], *[f"{pq[m][qid]:.4f}" for m in metrics],
                        f"{rerank_ms_by_q[qid]:.1f}", changed[qid]])
    return summary
