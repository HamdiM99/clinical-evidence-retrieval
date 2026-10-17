"""Core data model shared by ingestion, indexing, retrieval and evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""
    source: str = "unknown"          # e.g. "beir/nfcorpus", "has", "pubmed"
    lang: str = "en"
    url: str | None = None
    published: str | None = None     # ISO date
    evidence_grade: str | None = None  # e.g. "A", "B", "C", "AE" for guidelines
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    position: int
    text: str
    title: str = ""
    section: str | None = None
    start_char: int = 0
    end_char: int = 0
    source: str = "unknown"
    lang: str = "en"
    published: str | None = None
    evidence_grade: str | None = None

    def to_index_doc(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def embedding_text(self) -> str:
        """Text given to the embedding model: title prefix keeps the passage interpretable."""
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class Hit:
    """One chunk returned by a retriever."""
    chunk_id: str
    doc_id: str
    score: float
    text: str = ""
    title: str = ""


@dataclass
class DocResult:
    """One document after chunk aggregation and fusion."""
    doc_id: str
    score: float
    best_chunk_id: str | None = None
    best_text: str = ""
    best_title: str = ""
    ranks: dict[str, int] = field(default_factory=dict)  # rank in each retriever list (1-based)
