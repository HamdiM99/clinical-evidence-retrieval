"""Training pairs for a cross-encoder, with hard negatives mined by our own first stage.

For each training query:
- positives: judged relevant documents (qrels > 0), at most `max_pos` per query, those retrieved by the first
  stage first ("hard positives", the ones the re-ranker will actually see), then the others; a positive
  outside the candidates uses its document's first chunk, the same passage the index holds.
- negatives: first-stage candidates ranked `skip_top + 1 .. depth` that are not judged relevant,
  `neg_per_pos` per positive. `skip_top` protects against false negatives where judgments are incomplete:
  a top-ranked unjudged document is often relevant (NFCorpus), much less so when each query is linked to its
  source (SciFact).
- excluded queries: any query whose normalised text equals a test query (leakage), and held-out queries
  used to tune the re-ranker.

Everything is seeded: the same inputs give the same file.
"""
from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


def normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


@dataclass
class PairParams:
    max_pos: int = 8
    neg_per_pos: int = 4
    depth: int = 50
    skip_top: int = 0
    seed: int = 13


def build_pairs(records: Iterable[dict], qrels: Mapping[str, Mapping[str, int]],
                doc_passages: Mapping[str, str], params: PairParams, *,
                exclude_qids: set[str] = frozenset(), exclude_texts: set[str] = frozenset(),
                dataset: str = "") -> tuple[list[dict], dict]:
    """Return (pairs, stats). A pair is {"dataset", "qid", "query", "doc_id", "passage", "label"}."""
    from cer.rerank import passage_text

    rng = random.Random(params.seed)
    pairs: list[dict] = []
    stats = {"queries": 0, "skipped_excluded": 0, "skipped_leak": 0, "skipped_no_pos": 0,
             "positives": 0, "hard_positives": 0, "negatives": 0}
    for r in records:
        qid, query = r["qid"], r["query"]
        if qid in exclude_qids:
            stats["skipped_excluded"] += 1
            continue
        if normalise(query) in exclude_texts:
            stats["skipped_leak"] += 1
            continue
        rels = {d for d, v in qrels.get(qid, {}).items() if v > 0}
        cands = r["candidates"][: params.depth]
        cand_text = {c["doc_id"]: passage_text(c["title"], c["text"]) for c in cands}
        hard = [d for d in (c["doc_id"] for c in cands) if d in rels]
        other = sorted((rels - set(hard)) & set(doc_passages))
        rng.shuffle(other)
        pos = (hard + other)[: params.max_pos]
        if not pos:
            stats["skipped_no_pos"] += 1
            continue
        neg_pool = [c["doc_id"] for c in cands[params.skip_top:] if c["doc_id"] not in rels]
        neg = neg_pool[: params.neg_per_pos * len(pos)]  # hardest first: highest fused rank
        stats["queries"] += 1
        stats["positives"] += len(pos)
        stats["hard_positives"] += sum(1 for d in pos if d in cand_text)
        stats["negatives"] += len(neg)
        for d, label in [(d, 1.0) for d in pos] + [(d, 0.0) for d in neg]:
            pairs.append({"dataset": dataset, "qid": qid, "query": query, "doc_id": d,
                          "passage": cand_text.get(d) or doc_passages[d], "label": label})
    return pairs, stats
