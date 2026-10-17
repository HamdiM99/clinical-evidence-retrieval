"""Paired randomization (permutation) test between two systems on per-query scores."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def paired_permutation_test(a: Mapping[str, float], b: Mapping[str, float], n_permutations: int = 10_000,
                            seed: int = 0) -> dict[str, float]:
    """Two-sided test of H0: mean(a - b) = 0, by randomly flipping the sign of per-query differences."""
    qids = sorted(set(a) & set(b))
    if not qids:
        raise ValueError("no common queries")
    d = np.array([a[q] - b[q] for q in qids])
    observed = d.mean()
    if np.allclose(d, 0):
        return {"mean_a": float(np.mean([a[q] for q in qids])), "mean_b": float(np.mean([b[q] for q in qids])),
                "delta": 0.0, "p_value": 1.0, "n_queries": len(qids)}
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (n_permutations + 1)
    return {"mean_a": float(np.mean([a[q] for q in qids])), "mean_b": float(np.mean([b[q] for q in qids])),
            "delta": float(observed), "p_value": float(p), "n_queries": len(qids)}
