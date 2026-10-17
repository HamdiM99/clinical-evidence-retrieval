"""BEIR datasets: download and load (corpus.jsonl, queries.jsonl, qrels/<split>.tsv).

Format reference: https://github.com/beir-cellar/beir
"""
from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

from cer.schema import Document

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
MEDICAL_DATASETS = ("nfcorpus", "scifact", "trec-covid")


def download(name: str, out_dir: str | Path = "data") -> Path:
    """Download and unzip a BEIR dataset into out_dir/name. No-op if already present."""
    out_dir = Path(out_dir)
    target = out_dir / name
    if (target / "corpus.jsonl").exists():
        return target
    out_dir.mkdir(parents=True, exist_ok=True)
    url = BEIR_URL.format(name=name)
    with urllib.request.urlopen(url) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(out_dir)
    return target


def iter_corpus(dataset_dir: str | Path, source: str | None = None) -> Iterator[Document]:
    dataset_dir = Path(dataset_dir)
    source = source or f"beir/{dataset_dir.name}"
    with open(dataset_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            yield Document(
                doc_id=str(row["_id"]),
                title=row.get("title") or "",
                text=row.get("text") or "",
                source=source,
                lang="en",
                metadata={k: v for k, v in (row.get("metadata") or {}).items() if isinstance(v, (str, int, float))},
            )


def load_queries(dataset_dir: str | Path) -> dict[str, str]:
    out = {}
    with open(Path(dataset_dir) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[str(row["_id"])] = row["text"]
    return out


def load_qrels(dataset_dir: str | Path, split: str = "test") -> dict[str, dict[str, int]]:
    """{query_id: {doc_id: relevance}}. Header line is skipped."""
    qrels: dict[str, dict[str, int]] = {}
    with open(Path(dataset_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, rel = row[0], row[1], int(row[2])
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def load(dataset_dir: str | Path, split: str = "test"):
    """Return (queries restricted to the split, qrels). The corpus is streamed with iter_corpus."""
    qrels = load_qrels(dataset_dir, split)
    queries = {qid: q for qid, q in load_queries(dataset_dir).items() if qid in qrels}
    return queries, qrels
