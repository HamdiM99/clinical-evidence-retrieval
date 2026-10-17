"""In-process backend: BM25 (rank_bm25) + exact cosine search. Same interface as Elasticsearch.

Used by tests, CI and small experiments. Not meant for large corpora (exact search is O(N)).
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from cer.index import Filters
from cer.schema import Chunk, Hit

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _match(chunk: Chunk, filters: Filters | None) -> bool:
    if not filters:
        return True
    for field, value in filters.items():
        allowed = value if isinstance(value, list) else [value]
        if getattr(chunk, field, None) not in allowed:
            return False
    return True


class MemoryBackend:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []
        self.matrix: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None

    def create_index(self, dim: int, version: int = 1) -> str:
        self.__init__()
        return f"memory-v{version}"

    def index_chunks(self, chunks: Sequence[Chunk], vectors: np.ndarray, index_name: str = "") -> int:
        self.chunks.extend(chunks)
        if vectors is not None and len(vectors):
            self._vectors.append(np.asarray(vectors, dtype=np.float32))
        return len(chunks)

    def finalize(self, index_name: str = "") -> None:
        self.matrix = np.vstack(self._vectors) if self._vectors else None
        self.bm25 = BM25Okapi([tokenize(c.title + " " + c.text) for c in self.chunks])

    def swap_alias(self, index_name: str) -> None:  # nothing to swap in memory
        return None

    def count(self) -> int:
        return len(self.chunks)

    def _hits(self, scores: np.ndarray, k: int, filters: Filters | None) -> list[Hit]:
        order = np.argsort(-scores, kind="stable")
        out = []
        for i in order:
            c = self.chunks[i]
            if not _match(c, filters):
                continue
            out.append(Hit(chunk_id=c.chunk_id, doc_id=c.doc_id, score=float(scores[i]), text=c.text, title=c.title))
            if len(out) >= k:
                break
        return out

    def hybrid_search(self, text, vector, k_bm25, k_knn, num_candidates=0, filters=None):
        bm25_hits: list[Hit] = []
        knn_hits: list[Hit] = []
        if k_bm25 > 0 and self.bm25 is not None:
            scores = np.asarray(self.bm25.get_scores(tokenize(text)))
            bm25_hits = [h for h in self._hits(scores, k_bm25, filters) if h.score > 0]
        if k_knn > 0 and vector is not None and self.matrix is not None:
            knn_hits = self._hits(self.matrix @ np.asarray(vector, dtype=np.float32), k_knn, filters)
        return bm25_hits, knn_hits
