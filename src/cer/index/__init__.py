"""Search backends share one interface so retrieval and evaluation code never depend on the engine."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from cer.config import BackendConfig
from cer.schema import Chunk, Hit

Filters = dict[str, "str | list[str]"]  # keyword-field equality filters, e.g. {"lang": "en"}


class SearchBackend(Protocol):
    def create_index(self, dim: int, version: int) -> str: ...
    def index_chunks(self, chunks: Sequence[Chunk], vectors: np.ndarray, index_name: str) -> int: ...
    def finalize(self, index_name: str) -> None: ...
    def swap_alias(self, index_name: str) -> None: ...
    def hybrid_search(self, text: str, vector: np.ndarray | None, k_bm25: int, k_knn: int,
                      num_candidates: int, filters: Filters | None = None) -> tuple[list[Hit], list[Hit]]: ...
    def count(self) -> int: ...


def build_backend(cfg: BackendConfig) -> SearchBackend:
    if cfg.type == "memory":
        from cer.index.memory import MemoryBackend
        return MemoryBackend()
    from cer.index.es import ElasticsearchBackend
    return ElasticsearchBackend(cfg)
