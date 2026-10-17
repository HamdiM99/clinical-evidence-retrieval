import itertools

from cer.ingest.chunking import chunk_document
from cer.schema import Document


def long_doc(n_sent=40):
    text = " ".join(f"Sentence number {i} talks about clinical evidence and guidelines." for i in range(n_sent))
    return Document(doc_id="D", title="T", text=text)


def test_short_document_is_one_chunk():
    d = Document(doc_id="a", text="Metformin is first line.")
    cs = chunk_document(d, max_words=50, overlap_words=10)
    assert len(cs) == 1 and cs[0].text == d.text and (cs[0].start_char, cs[0].end_char) == (0, len(d.text))


def test_long_document_windows_respect_size_and_offsets():
    d = long_doc()
    cs = chunk_document(d, max_words=60, overlap_words=15)
    assert len(cs) > 1
    for c in cs:
        assert len(c.text.split()) <= 60
        assert d.text[c.start_char:c.end_char] == c.text      # citable offsets
    assert cs[0].start_char == 0 and cs[-1].end_char == len(d.text)  # full coverage


def test_consecutive_chunks_overlap():
    cs = chunk_document(long_doc(), max_words=60, overlap_words=15)
    for a, b in itertools.pairwise(cs):
        assert b.start_char < a.end_char        # overlap
        assert b.start_char > a.start_char      # progress


def test_ids_are_deterministic_and_unique():
    a = chunk_document(long_doc(), max_words=60, overlap_words=15)
    b = chunk_document(long_doc(), max_words=60, overlap_words=15)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_giant_sentence_is_split():
    d = Document(doc_id="g", text=" ".join(["word"] * 500))
    cs = chunk_document(d, max_words=100, overlap_words=0)
    assert all(len(c.text.split()) <= 100 for c in cs) and len(cs) == 5
