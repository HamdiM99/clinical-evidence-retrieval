"""Build the cross-encoder training and validation files from exported first-stage candidates.

Inputs (produced by `cer download` and `cer candidates ... --split train --split dev --split test`):
  runs/nfcorpus-hybrid/candidates-{train,dev}.jsonl, runs/scifact-hybrid/candidates-train.jsonl
Outputs (derived from BEIR, kept out of git):
  data/rerank-train/train.jsonl   {"dataset", "qid", "query", "doc_id", "passage", "label"}, shuffled
  data/rerank-train/valid.jsonl   validation queries with their first-stage top `valid_top_n` and qrels
  data/rerank-train/stats.json    parameters, counts, exclusions

Validation = the re-ranker tuning sets of step 3: NFCorpus dev and the first 200 SciFact train queries,
which are therefore excluded from training. Test queries are never read except to remove train queries
whose text is identical to a test query.

Usage: python scripts/build_rerank_train.py [--out data/rerank-train]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from cer.data import beir
from cer.ingest.chunking import chunk_document
from cer.rerank import passage_text
from cer.rerank.candidates import read_candidates
from cer.rerank.training import PairParams, build_pairs, normalise

SCIFACT_HOLDOUT = 200


def first_chunk_passages(dataset_dir: Path, max_words: int, overlap_words: int) -> dict[str, str]:
    out = {}
    for doc in beir.iter_corpus(dataset_dir):
        chunks = chunk_document(doc, max_words=max_words, overlap_words=overlap_words)
        if chunks:
            out[doc.doc_id] = passage_text(chunks[0].title, chunks[0].text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="data/rerank-train")
    ap.add_argument("--max-pos", type=int, default=8)
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--nfcorpus-skip-top", type=int, default=3)
    ap.add_argument("--scifact-repeat", type=int, default=3, help="repeat SciFact pairs to balance the datasets")
    ap.add_argument("--valid-top-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nf_train = list(read_candidates(runs / "nfcorpus-hybrid/candidates-train.jsonl"))
    nf_dev = list(read_candidates(runs / "nfcorpus-hybrid/candidates-dev.jsonl"))
    sf_train = list(read_candidates(runs / "scifact-hybrid/candidates-train.jsonl"))
    sf_holdout, sf_fit = sf_train[:SCIFACT_HOLDOUT], sf_train[SCIFACT_HOLDOUT:]  # same order as `--limit 200`

    # leakage guard: texts of every test query, plus validation queries
    exclude_texts = {normalise(t) for ds in ("nfcorpus", "scifact")
                     for t in beir.load(Path("data") / ds, "test")[0].values()}
    exclude_texts |= {normalise(r["query"]) for r in nf_dev + sf_holdout}

    pairs, stats = [], {}
    for ds, records, skip_top, repeat in [("nfcorpus", nf_train, args.nfcorpus_skip_top, 1),
                                          ("scifact", sf_fit, 0, args.scifact_repeat)]:
        meta = json.loads((runs / f"{ds}-hybrid/candidates-train.meta.json").read_text(encoding="utf-8"))
        ch = meta["config"]["chunking"]
        docs = first_chunk_passages(Path("data") / ds, ch["max_words"], ch["overlap_words"])
        params = PairParams(max_pos=args.max_pos, neg_per_pos=args.neg_per_pos, depth=args.depth,
                            skip_top=skip_top, seed=args.seed)
        p, s = build_pairs(records, beir.load_qrels(Path("data") / ds, "train"), docs, params,
                           exclude_texts=exclude_texts, dataset=ds)
        pairs += p * repeat
        stats[ds] = {**s, "pairs": len(p), "repeat": repeat, "params": params.__dict__}
    random.Random(args.seed).shuffle(pairs)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in pairs)

    with open(out / "valid.jsonl", "w", encoding="utf-8") as f:
        for ds, split, records in [("nfcorpus", "dev", nf_dev), ("scifact", "train", sf_holdout)]:
            qrels = beir.load_qrels(Path("data") / ds, split)
            for r in records:
                cands = [{"doc_id": c["doc_id"], "passage": passage_text(c["title"], c["text"])}
                         for c in r["candidates"][: args.valid_top_n]]
                f.write(json.dumps({"dataset": ds, "qid": r["qid"], "query": r["query"], "candidates": cands,
                                    "qrels": qrels.get(r["qid"], {})}, ensure_ascii=False) + "\n")

    summary = {"train_pairs": len(pairs), "positives_share": round(sum(x["label"] for x in pairs) / len(pairs), 3),
               "valid_queries": len(nf_dev) + len(sf_holdout), "valid_top_n": args.valid_top_n,
               "datasets": stats, "seed": args.seed, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
