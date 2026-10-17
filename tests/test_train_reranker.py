import importlib.util
import random
from pathlib import Path

import pytest

from cer.eval.metrics import ndcg_at_k

pytest.importorskip("torch")
pytest.importorskip("transformers")

spec = importlib.util.spec_from_file_location(
    "train_reranker", Path(__file__).parents[1] / "scripts" / "train_reranker.py")
train_reranker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_reranker)


def test_self_contained_ndcg_matches_cer_metrics():
    rng = random.Random(0)
    for _ in range(200):
        docs = [f"d{i}" for i in range(30)]
        rng.shuffle(docs)
        rels = {d: rng.choice([0, 1, 2]) for d in rng.sample(docs, 8)}
        assert train_reranker.ndcg_at_10(docs, rels) == pytest.approx(ndcg_at_k(docs, rels, 10))


def test_first_stage_order_is_candidate_order():
    valid = [{"dataset": "a", "query": "q", "qrels": {"x": 1},
              "candidates": [{"doc_id": "x", "passage": ""}, {"doc_id": "y", "passage": ""}]},
             {"dataset": "b", "query": "q", "qrels": {"x": 1},
              "candidates": [{"doc_id": "y", "passage": ""}, {"doc_id": "x", "passage": ""}]}]
    res = train_reranker.first_stage(valid)
    assert res["ndcg@10/a"] == 1.0 and res["ndcg@10/b"] == pytest.approx(1 / 1.5849625)
    assert res["ndcg@10/mean"] == pytest.approx((1.0 + res["ndcg@10/b"]) / 2)
