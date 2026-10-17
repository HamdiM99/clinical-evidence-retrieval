"""Sentence-aware chunking with word overlap and character offsets.

- Short documents (<= max_words) stay one chunk.
- Long documents are split on sentence boundaries into windows of ~max_words, consecutive windows
  sharing ~overlap_words words, so that a fact sitting on a boundary is fully present in one chunk.
- Offsets (start_char, end_char) point into the original text: results stay citable.
- chunk_id is deterministic: re-ingesting the same document gives the same ids (idempotent upserts).
"""
from __future__ import annotations

import hashlib
import re

from cer.schema import Chunk, Document

CHUNKER_VERSION = "s1"
_SENT = re.compile(r"[^.!?;\n]+(?:[.!?;]+|\n+|$)")


def _sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans (start, end) in text; whitespace-only spans are dropped."""
    spans = []
    for m in _SENT.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((s, e))
    return spans


def _n_words(text: str) -> int:
    return len(text.split())


def chunk_id(doc_id: str, position: int) -> str:
    return hashlib.sha1(f"{doc_id}|{CHUNKER_VERSION}|{position}".encode()).hexdigest()[:20]


def chunk_document(doc: Document, max_words: int = 250, overlap_words: int = 40) -> list[Chunk]:
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    text = doc.text or ""
    base = {"doc_id": doc.doc_id, "title": doc.title, "source": doc.source, "lang": doc.lang,
                "published": doc.published, "evidence_grade": doc.evidence_grade}

    if _n_words(text) <= max_words:
        return [Chunk(chunk_id=chunk_id(doc.doc_id, 0), position=0, text=text,
                      start_char=0, end_char=len(text), **base)]

    sents = _sentences(text)
    # a single "sentence" longer than max_words is split on words
    units: list[tuple[int, int]] = []
    for s, e in sents:
        words = list(re.finditer(r"\S+", text[s:e]))
        if len(words) <= max_words:
            units.append((s, e))
        else:
            for i in range(0, len(words), max_words):
                block = words[i:i + max_words]
                units.append((s + block[0].start(), s + block[-1].end()))

    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        j, words = i, 0
        while j < len(units) and (words + _n_words(text[units[j][0]:units[j][1]]) <= max_words or j == i):
            words += _n_words(text[units[j][0]:units[j][1]])
            j += 1
        start, end = units[i][0], units[j - 1][1]
        pos = len(chunks)
        chunks.append(Chunk(chunk_id=chunk_id(doc.doc_id, pos), position=pos, text=text[start:end],
                            start_char=start, end_char=end, **base))
        if j >= len(units):
            break
        # step back so the next chunk starts with ~overlap_words words of this one
        back, k = 0, j
        while k - 1 > i and back + _n_words(text[units[k - 1][0]:units[k - 1][1]]) <= overlap_words:
            k -= 1
            back += _n_words(text[units[k][0]:units[k][1]])
        i = k if k > i else j
    return chunks
