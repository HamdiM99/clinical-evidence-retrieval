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


def pair(ds, qid, doc, label):
    return {"dataset": ds, "qid": qid, "query": f"query {qid}", "doc_id": doc, "passage": doc, "label": label}


def test_listwise_groups_positive_first_negatives_of_same_query():
    train = [pair("a", "q1", "p1", 1.0), pair("a", "q1", "p2", 1.0),
             *[pair("a", "q1", f"n{i}", 0.0) for i in range(10)],
             pair("b", "q1", "p3", 1.0), pair("b", "q1", "m1", 0.0),    # same qid, other dataset
             pair("a", "q2", "p4", 1.0)]                               # no negative: skipped
    groups = train_reranker.make_groups(train, 4, random.Random(0))
    assert len(groups) == 3 and all(len(g) == 4 for g in groups)
    for g in groups:
        assert g[0]["label"] == 1.0 and all(x["label"] == 0.0 for x in g[1:])
        assert {(x["dataset"], x["qid"]) for x in g} == {(g[0]["dataset"], g[0]["qid"])}
    g_b = next(g for g in groups if g[0]["dataset"] == "b")
    assert [x["doc_id"] for x in g_b[1:]] == ["m1"] * 3                # too few negatives: with replacement
    g_a = [g for g in groups if g[0]["dataset"] == "a"]
    assert all(len({x["doc_id"] for x in g[1:]}) == 3 for g in g_a)    # enough negatives: distinct


def test_listwise_batches_hold_whole_groups():
    class Args:
        loss, group_size, batch_size = "listwise", 4, 8
    train = [pair("a", f"q{q}", f"p{q}", 1.0) for q in range(5)]
    train += [pair("a", f"q{q}", f"n{q}{i}", 0.0) for q in range(5) for i in range(5)]
    bs = train_reranker.batches(train, Args, random.Random(0))
    assert [len(b) for b in bs] == [8, 8, 4]
    assert all(b[i]["label"] == 1.0 for b in bs for i in range(0, len(b), 4))
