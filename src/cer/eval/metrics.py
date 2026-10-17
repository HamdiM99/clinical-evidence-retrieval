"""Ranking metrics, TREC conventions (same definitions as trec_eval / ranx defaults).

- nDCG@k with linear gains (gain = relevance), log2 discount; ideal ranking from the qrels.
- Recall@k, MRR@k, MAP@k with "relevant" = relevance > 0.
- Ties in scores are broken by doc id (descending score, then ascending id) for determinism.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

Run = Mapping[str, Mapping[str, float]]     # {qid: {doc_id: score}}
Qrels = Mapping[str, Mapping[str, int]]     # {qid: {doc_id: relevance}}


def ranked(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda d: (-scores[d], d))


def ndcg_at_k(ranking: list[str], rels: Mapping[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: list[str], rels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    return len(relevant & set(ranking[:k])) / len(relevant) if relevant else 0.0


def mrr_at_k(ranking: list[str], rels: Mapping[str, int], k: int) -> float:
    for i, d in enumerate(ranking[:k], start=1):
        if rels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ap_at_k(ranking: list[str], rels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    hits, s = 0, 0.0
    for i, d in enumerate(ranking[:k], start=1):
        if d in relevant:
            hits += 1
            s += hits / i
    return s / len(relevant)


_FUNCS = {"ndcg": ndcg_at_k, "recall": recall_at_k, "mrr": mrr_at_k, "map": ap_at_k}


def parse_metric(name: str) -> tuple[str, int]:
    base, _, k = name.partition("@")
    if base not in _FUNCS or not k.isdigit():
        raise ValueError(f"unknown metric {name!r}; use e.g. ndcg@10, recall@100, mrr@10, map@100")
    return base, int(k)


def per_query(run: Run, qrels: Qrels, metrics: list[str]) -> dict[str, dict[str, float]]:
    """{metric: {qid: value}} over the queries present in qrels (missing runs score 0)."""
    out: dict[str, dict[str, float]] = {m: {} for m in metrics}
    for qid, rels in qrels.items():
        if not any(r > 0 for r in rels.values()):
            continue
        ranking = ranked(run.get(qid, {}))
        for m in metrics:
            base, k = parse_metric(m)
            out[m][qid] = _FUNCS[base](ranking, rels, k)
    return out


def evaluate(run: Run, qrels: Qrels, metrics: list[str]) -> dict[str, float]:
    pq = per_query(run, qrels, metrics)
    return {m: (sum(v.values()) / len(v) if v else 0.0) for m, v in pq.items()}
