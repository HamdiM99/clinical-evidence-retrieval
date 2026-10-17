import json

from cer.config import ExperimentConfig
from cer.eval.runner import read_trec, run_experiment
from cer.rerank.candidates import candidates_to_run, export_candidates, read_candidates


def cfg(tmp_path, mini_dir):
    return ExperimentConfig(
        name="mini-hybrid", dataset="mini", data_dir=str(mini_dir.parent),
        backend={"type": "memory"}, embedder={"type": "hashing", "dim": 256, "batch_size": 4},
        chunking={"max_words": 80, "overlap_words": 10},
        retrieval={"mode": "hybrid", "k_bm25": 20, "k_knn": 20, "top_k": 10},
        eval={"out_dir": str(tmp_path)})


def test_candidates_replay_the_first_stage_exactly(tmp_path, mini_dir):
    c = cfg(tmp_path, mini_dir)
    info = export_candidates(c, "test", progress=False)
    records = list(read_candidates(info["path"]))
    assert [r["qid"] for r in records] == ["q1", "q2", "q3", "q4", "q5"] and info["n_queries"] == 5
    for r in records:
        assert [x["rank"] for x in r["candidates"]] == list(range(1, len(r["candidates"]) + 1))
        assert len(r["candidates"]) <= 10 and all(x["text"] and x["chunk_id"] for x in r["candidates"])
        assert len({x["doc_id"] for x in r["candidates"]}) == len(r["candidates"])  # one entry per document
    # same scores as the run written by `cer eval` with the same config
    run_experiment(c, progress=False)
    expected = read_trec(tmp_path / "mini-hybrid" / "run.trec")
    replay = candidates_to_run(records)
    assert replay.keys() == expected.keys()
    for qid in expected:
        assert replay[qid].keys() == expected[qid].keys()
        assert all(abs(replay[qid][d] - expected[qid][d]) < 1e-6 for d in expected[qid])
    meta = json.loads((tmp_path / "mini-hybrid" / "candidates-test.meta.json").read_text(encoding="utf-8"))
    assert meta["split"] == "test" and meta["config"]["retrieval"]["mode"] == "hybrid"
    assert meta["latency_ms"]["total"]["p50"] >= 0


def test_candidates_titles_and_limit(tmp_path, mini_dir):
    info = export_candidates(cfg(tmp_path, mini_dir), "test", progress=False, limit=2)
    records = list(read_candidates(info["path"]))
    assert len(records) == 2
    assert any(x["title"] for r in records for x in r["candidates"])
