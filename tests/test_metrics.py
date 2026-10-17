import random

import pytest

from cer.eval.metrics import evaluate, ndcg_at_k, parse_metric

ranx = pytest.importorskip("ranx")


def random_case(seed):
    rnd = random.Random(seed)
    docs = [f"d{i}" for i in range(40)]
    qrels, run = {}, {}
    for q in range(25):
        qid = f"q{q}"
        rel = rnd.sample(docs, rnd.randint(1, 6))
        qrels[qid] = {d: rnd.choice([1, 2]) for d in rel}
        qrels[qid][rnd.choice([d for d in docs if d not in rel])] = 0   # judged non-relevant
        scores = rnd.sample(range(1000), 30)          # distinct scores: no ties
        run[qid] = {d: float(s) for d, s in zip(rnd.sample(docs, 30), scores)}
    return qrels, run


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_ranx(seed):
    qrels, run = random_case(seed)
    metrics = ["ndcg@10", "recall@20", "mrr@10", "map@100"]
    ours = evaluate(run, qrels, metrics)
    theirs = ranx.evaluate(ranx.Qrels(qrels), ranx.Run(run), metrics)
    for m in metrics:
        assert ours[m] == pytest.approx(theirs[m], abs=1e-6), m


def test_ndcg_perfect_and_zero():
    rels = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rels, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], rels, 10) == 0.0


def test_bad_metric_name():
    with pytest.raises(ValueError):
        parse_metric("ndcg")
