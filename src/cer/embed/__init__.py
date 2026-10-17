"""Embedders: one interface, two backends.

- SentenceTransformerEmbedder: real models (e5, bge, ...), with the query/passage prefixes they expect.
- HashingEmbedder: deterministic bag-of-words hashing. No download, used in tests and CI only.
"""
from __future__ import annotations

import hashlib
import itertools
import re
from typing import Protocol

import numpy as np

from cer.config import EmbedderConfig


class Embedder(Protocol):
    dim: int

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


def _default_prefixes(model: str) -> tuple[str, str]:
    m = model.lower()
    if "e5" in m:
        return "query: ", "passage: "
    if "bge" in m and "m3" not in m:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbedderConfig):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model, device=cfg.device)
        if cfg.max_seq_length:
            self.model.max_seq_length = cfg.max_seq_length
        qp, pp = _default_prefixes(cfg.model)
        self.query_prefix = cfg.query_prefix if cfg.query_prefix is not None else qp
        self.passage_prefix = cfg.passage_prefix if cfg.passage_prefix is not None else pp
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.cfg.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])


_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """Signed feature hashing of lowercase tokens + bigrams, L2-normalised. Test-only."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        feats = toks + [a + "_" + b for a, b in itertools.pairwise(toks)]
        for t in feats:
            h = int.from_bytes(hashlib.md5(t.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    embed_passages = embed_queries


def build_embedder(cfg: EmbedderConfig) -> Embedder:
    if cfg.type == "hashing":
        return HashingEmbedder(cfg.dim)
    return SentenceTransformerEmbedder(cfg)
