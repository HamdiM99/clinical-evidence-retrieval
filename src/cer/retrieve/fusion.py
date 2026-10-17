"""Chunk -> document aggregation and rank fusion. Pure functions, unit-tested."""
from __future__ import annotations

from collections import defaultdict

from cer.schema import Hit


def aggregate_max(hits: list[Hit]) -> list[tuple[str, Hit]]:
    """Documents ordered by their best chunk. `hits` must be sorted best first.
    A document with many average chunks does not outrank one with a single excellent chunk."""
    seen: dict[str, Hit] = {}
    for h in hits:
        if h.doc_id not in seen:
            seen[h.doc_id] = h
    return list(seen.items())


def rrf(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> dict[str, float]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum_i w_i / (k + rank_i(d)), ranks from 1.
    Items are expected to be unique within each ranking (aggregate chunks first)."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    fused: dict[str, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        if w == 0:
            continue
        for rank, item in enumerate(ranking, start=1):
            fused[item] += w / (k + rank)
    return dict(fused)
