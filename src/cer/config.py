"""Experiment configuration (YAML) validated with pydantic.

One config = one reproducible run: dataset + backend/index version + embedder + chunking + retrieval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    type: Literal["memory", "elasticsearch"] = "memory"
    url: str = "http://localhost:9200"
    alias: str = "cer-chunks"
    index_version: int = 1
    request_timeout: int = 60
    bm25_fields: list[str] = Field(default_factory=lambda: ["title^2", "text", "text.raw"])


class EmbedderConfig(BaseModel):
    type: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    model: str = "intfloat/multilingual-e5-base"
    batch_size: int = 64
    device: str | None = None          # "cuda", "cpu", "mps" or auto
    dim: int = 384                      # only used by the hashing embedder
    query_prefix: str | None = None     # auto-detected from the model name if None
    passage_prefix: str | None = None
    max_seq_length: int | None = 512


class ChunkingConfig(BaseModel):
    max_words: int = 250
    overlap_words: int = 40


class RetrievalConfig(BaseModel):
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    k_bm25: int = 100
    k_knn: int = 100
    num_candidates: int = 200          # HNSW breadth (Elasticsearch)
    rrf_k: int = 60
    weights: tuple[float, float] = (1.0, 1.0)   # (bm25, dense)
    top_k: int = 100                   # documents returned after fusion


class RerankConfig(BaseModel):
    type: Literal["cross-encoder", "onnx"] = "cross-encoder"   # onnx: directory with model.onnx + tokenizer
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 50                    # fused documents re-scored; the rest keep their fused order
    max_length: int = 256              # tokens for (query, passage); 512 roughly doubles the cost
    batch_size: int = 16
    device: str | None = None
    threads: int | None = None         # CPU threads (ONNX Runtime); None = runtime default


class EvalConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["ndcg@10", "recall@100", "mrr@10", "map@100"])
    out_dir: str = "runs"


class ExperimentConfig(BaseModel):
    name: str
    dataset: str = "nfcorpus"
    data_dir: str = "data"
    split: str = "test"
    backend: BackendConfig = Field(default_factory=BackendConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig | None = None
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @property
    def dataset_dir(self) -> Path:
        return Path(self.data_dir) / self.dataset

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(data)
