"""Streaming ingestion: documents -> chunks -> embeddings (batched) -> bulk index -> finalize -> alias swap.

Memory stays bounded by the batch size, so the same code runs on 10^3 or 10^7 documents.
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from cer.embed import Embedder
from cer.index import SearchBackend
from cer.ingest.chunking import chunk_document
from cer.schema import Chunk, Document


@dataclass
class IngestReport:
    index_name: str
    documents: int
    chunks: int
    seconds: float
    embed_seconds: float
    index_seconds: float

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.seconds if self.seconds else 0.0


def _batches(chunks: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    batch: list[Chunk] = []
    for c in chunks:
        batch.append(c)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest(docs: Iterable[Document], backend: SearchBackend, embedder: Embedder | None, *,
           version: int = 1, max_words: int = 250, overlap_words: int = 40, batch_size: int = 256,
           swap_alias: bool = True, progress: bool = True) -> IngestReport:
    t0 = time.perf_counter()
    dim = embedder.dim if embedder is not None else 1
    index_name = backend.create_index(dim=dim, version=version)
    n_docs = n_chunks = 0
    t_embed = t_index = 0.0

    def chunk_stream():
        nonlocal n_docs
        for d in docs:
            n_docs += 1
            yield from chunk_document(d, max_words=max_words, overlap_words=overlap_words)

    for batch in tqdm(_batches(chunk_stream(), batch_size), disable=not progress, desc="ingest", unit="batch"):
        t1 = time.perf_counter()
        vectors = (embedder.embed_passages([c.embedding_text() for c in batch]) if embedder is not None
                   else np.zeros((len(batch), 1), dtype=np.float32))
        t2 = time.perf_counter()
        n_chunks += backend.index_chunks(batch, vectors, index_name)
        t3 = time.perf_counter()
        t_embed += t2 - t1
        t_index += t3 - t2

    backend.finalize(index_name)
    if swap_alias:
        backend.swap_alias(index_name)
    return IngestReport(index_name, n_docs, n_chunks, time.perf_counter() - t0, t_embed, t_index)
