import json

from cer.config import ExperimentConfig
from cer.data import beir
from cer.eval.runner import read_trec, run_experiment


def test_loader(mini_dir):
    queries, qrels = beir.load(mini_dir, "test")
    assert set(queries) == {"q1", "q2", "q3", "q4", "q5"}
    assert qrels["q1"] == {"d1": 2, "d2": 1, "d9": 1}
    docs = list(beir.iter_corpus(mini_dir))
    assert len(docs) == 12 and docs[0].source == "beir/mini"


def cfg(tmp_path, mini_dir, mode):
    return ExperimentConfig(
        name=f"mini-{mode}", dataset="mini", data_dir=str(mini_dir.parent),
        backend={"type": "memory"}, embedder={"type": "hashing", "dim": 256, "batch_size": 4},
        chunking={"max_words": 80, "overlap_words": 10},
        retrieval={"mode": mode, "k_bm25": 20, "k_knn": 20, "top_k": 10},
        eval={"out_dir": str(tmp_path)})


def test_end_to_end_memory(tmp_path, mini_dir):
    results = {m: run_experiment(cfg(tmp_path, mini_dir, m), progress=False) for m in ("bm25", "dense", "hybrid")}
    for m, s in results.items():
        assert 0.0 < s["metrics"]["ndcg@10"] <= 1.0, m
        assert s["n_queries"] == 5
        out = tmp_path / f"mini-{m}"
        assert {p.name for p in out.iterdir()} == {"config.yaml", "run.trec", "metrics.json", "per_query.csv"}
        assert json.loads((out / "metrics.json").read_text())["latency_ms"]["total"]["p95"] >= 0
        assert set(read_trec(out / "run.trec")) == {"q1", "q2", "q3", "q4", "q5"}
    # the long sleep review is split in several chunks but appears once per query (chunk -> doc aggregation)
    run = read_trec(tmp_path / "mini-hybrid" / "run.trec")
    assert all(list(r).count("d11") <= 1 for r in run.values())
    assert results["bm25"]["index"]["chunks"] > 12


def test_readme_block_replaced_between_markers():
    from cer.cli import README_END, README_START, replace_block

    text = f"intro\n{README_START}\nold table\n{README_END}\nrest\n"
    assert replace_block(text, "| new |") == f"intro\n{README_START}\n| new |\n{README_END}\nrest\n"
