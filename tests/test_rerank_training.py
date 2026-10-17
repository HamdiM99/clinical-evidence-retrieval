from cer.rerank.training import PairParams, build_pairs, normalise


def record(qid, query, doc_ids):
    return {"qid": qid, "query": query,
            "candidates": [{"doc_id": d, "title": "", "text": f"text of {d}"} for d in doc_ids]}


CANDS = [f"d{i}" for i in range(1, 21)]          # fused order d1 .. d20
QRELS = {"q1": {"d2": 1, "d5": 1, "x1": 1, "x2": 1, "d7": 0}}
DOCS = {"x1": "corpus x1", "x2": "corpus x2"}     # relevant but not retrieved


def test_positives_hard_first_then_corpus_and_capped():
    pairs, stats = build_pairs([record("q1", "q", CANDS)], QRELS, DOCS, PairParams(max_pos=3, neg_per_pos=1))
    pos = [p for p in pairs if p["label"] == 1.0]
    assert [p["doc_id"] for p in pos[:2]] == ["d2", "d5"]               # retrieved positives first
    assert len(pos) == 3 and pos[2]["doc_id"] in {"x1", "x2"}
    assert pos[2]["passage"] == DOCS[pos[2]["doc_id"]] and pos[0]["passage"] == "text of d2"
    assert stats["hard_positives"] == 2 and stats["positives"] == 3


def test_negatives_are_hardest_unjudged_or_irrelevant_after_skip_top():
    pairs, _ = build_pairs([record("q1", "q", CANDS)], QRELS, DOCS,
                           PairParams(max_pos=2, neg_per_pos=2, skip_top=3, depth=10))
    neg = [p["doc_id"] for p in pairs if p["label"] == 0.0]
    assert neg == ["d4", "d6", "d7", "d8"]   # ranks 1-3 skipped, positives d2/d5 never negatives, d7 (rel 0) kept


def test_exclusions_and_no_positive():
    recs = [record("q1", "Are Avocados Good for You?", CANDS), record("q2", "other", CANDS),
            record("q3", "kept", CANDS)]
    qrels = {**QRELS, "q3": {"d1": 1}}
    pairs, stats = build_pairs(recs, qrels, DOCS, PairParams(), exclude_texts={normalise("are avocados good for you")},
                               exclude_qids={"q2"})
    assert {p["qid"] for p in pairs} == {"q3"}
    assert stats["skipped_leak"] == 1 and stats["skipped_excluded"] == 1 and stats["queries"] == 1
    _, stats = build_pairs([record("q9", "no qrels", CANDS)], {}, DOCS, PairParams())
    assert stats["skipped_no_pos"] == 1


def test_deterministic_with_seed():
    a, _ = build_pairs([record("q1", "q", CANDS)], QRELS, DOCS, PairParams(max_pos=3, seed=1))
    b, _ = build_pairs([record("q1", "q", CANDS)], QRELS, DOCS, PairParams(max_pos=3, seed=1))
    assert a == b
