import json

import pytest

from cer.config import ExperimentConfig, RerankConfig
from cer.data import beir
from cer.rerank import passage_text
from cer.rerank.candidates import export_candidates, read_candidates
from cer.rerank.replay import cache_path, read_cache, rerank_order, rerank_run, score_candidates


def test_rerank_order_head_sorted_tail_kept():
    fused = ["a", "b", "c", "d", "e"]
    scores = {"a": 0.1, "b": 0.9, "c": 0.9}  # b and c tie: fused order breaks the tie
    assert rerank_order(fused, scores, top_n=3) == ["b", "c", "a", "d", "e"]
    assert rerank_order(fused, {d: 1.0 for d in fused}, top_n=5) == fused


@pytest.fixture
def candidates(tmp_path, mini_dir):
    cfg = ExperimentConfig(
        name="mini-hybrid", dataset="mini", data_dir=str(mini_dir.parent),
        backend={"type": "memory"}, embedder={"type": "hashing", "dim": 256, "batch_size": 4},
        chunking={"max_words": 80, "overlap_words": 10},
        retrieval={"mode": "hybrid", "k_bm25": 20, "k_knn": 20, "top_k": 10},
        eval={"out_dir": str(tmp_path)})
    return export_candidates(cfg, "test", progress=False)


class OracleReranker:
    """Scores a passage with the qrels relevance of its document: an upper bound for re-ranking."""
    name = "oracle"

    def __init__(self, candidates_file, qrels):
        self.rel = {}
        for r in read_candidates(candidates_file):
            for c in r["candidates"]:
                self.rel[(r["query"], passage_text(c["title"], c["text"]))] = qrels[r["qid"]].get(c["doc_id"], 0)
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        return [float(self.rel[(query, p)]) for p in passages]


class FailingReranker:
    name = "failing"

    def score(self, query, passages):
        raise AssertionError("cache should have been reused")


def test_oracle_rerank_improves_ndcg_and_keeps_recall(tmp_path, mini_dir, candidates):
    path = tmp_path / "mini-hybrid" / "candidates-test.jsonl"
    qrels = beir.load_qrels(mini_dir, "test")
    cfg = RerankConfig(model="test/oracle", top_n=10)
    oracle = OracleReranker(path, qrels)
    cache = score_candidates(path, oracle, cfg, depth=10, progress=False)
    assert cache == cache_path(path, cfg) and oracle.calls == 5 + 1  # + warm-up
    assert all(len(r["scores"]) == len(r["doc_ids"]) for r in read_cache(cache).values())
    score_candidates(path, FailingReranker(), cfg, depth=10, progress=False)  # deep enough: reused

    first = json.loads((tmp_path / "mini-hybrid" / "candidates-test.meta.json").read_text(encoding="utf-8"))
    assert first["n_queries"] == 5
    s10 = rerank_run(path, cfg, out_dir=tmp_path)
    s3 = rerank_run(path, cfg.model_copy(update={"top_n": 3}), out_dir=tmp_path)
    assert s10["name"] == "mini-hybrid-oracle-L256-top10" and s10["split"] == "test"
    from cer.eval.metrics import per_query
    from cer.rerank.candidates import candidates_to_run
    fused = per_query(candidates_to_run(read_candidates(path)), qrels, ["ndcg@10", "recall@100"])
    fused_ndcg = sum(fused["ndcg@10"].values()) / 5
    assert s10["metrics"]["ndcg@10"] >= s3["metrics"]["ndcg@10"] >= fused_ndcg
    assert s10["metrics"]["ndcg@10"] > fused_ndcg
    assert s10["metrics"]["recall@100"] == pytest.approx(sum(fused["recall@100"].values()) / 5)
    assert s3["rerank_latency"].startswith("estimated") and "first_stage" in s10["latency_ms"]
    out = tmp_path / s10["name"]
    assert {p.name for p in out.iterdir()} == {"config.yaml", "run.trec", "metrics.json", "per_query.csv"}


def test_rerank_run_refuses_a_shallow_cache(tmp_path, mini_dir, candidates):
    path = tmp_path / "mini-hybrid" / "candidates-test.jsonl"
    cfg = RerankConfig(model="test/oracle", top_n=3)
    score_candidates(path, OracleReranker(path, beir.load_qrels(mini_dir, "test")), cfg, depth=3, progress=False)
    with pytest.raises(ValueError, match="depth"):
        rerank_run(path, cfg.model_copy(update={"top_n": 10}), out_dir=tmp_path)
