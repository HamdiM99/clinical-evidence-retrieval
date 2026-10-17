"""Second stage: re-rank first-stage candidates (exported once, replayed offline).

A reranker scores (query, passage) pairs; higher is more relevant. Scores of different queries are not
comparable, only the order within one query matters.
"""
from __future__ import annotations

from typing import Protocol

from cer.config import RerankConfig


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


def passage_text(title: str, text: str) -> str:
    """Same title prefix as the embedding text, so the passage stays interpretable out of context."""
    return f"{title}. {text}" if title else text


class CrossEncoderReranker:
    def __init__(self, cfg: RerankConfig):
        from sentence_transformers import CrossEncoder  # optional dependency

        self.cfg = cfg
        self.name = cfg.model
        self.model = CrossEncoder(cfg.model, max_length=cfg.max_length, device=cfg.device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], batch_size=self.cfg.batch_size,
                                    show_progress_bar=False, convert_to_numpy=True)
        return [float(s) for s in scores]


def build_reranker(cfg: RerankConfig) -> Reranker:
    if cfg.type == "onnx":
        from cer.rerank.onnx import OnnxReranker
        return OnnxReranker(cfg)
    return CrossEncoderReranker(cfg)
