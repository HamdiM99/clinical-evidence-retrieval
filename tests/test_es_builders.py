import pytest

from cer.config import BackendConfig
from cer.index.es import (
    ElasticsearchBackend,
    build_bm25_query,
    build_index_body,
    build_knn_query,
    build_msearch,
    parse_hits,
)


def test_index_body():
    body = build_index_body(dim=384)
    emb = body["mappings"]["properties"]["embedding"]
    assert emb["dims"] == 384 and emb["index_options"]["type"] == "int8_hnsw" and emb["similarity"] == "cosine"
    assert body["settings"]["refresh_interval"] == "-1"          # bulk-load mode
    assert body["mappings"]["dynamic"] == "strict"


def test_queries_with_filters():
    q = build_bm25_query("aspirin", 50, ["title^2", "text"], {"lang": "en", "source": ["has", "pubmed"]})
    assert q["size"] == 50
    assert {"term": {"lang": "en"}} in q["query"]["bool"]["filter"]
    assert {"terms": {"source": ["has", "pubmed"]}} in q["query"]["bool"]["filter"]
    k = build_knn_query([0.1, 0.2], k=10, num_candidates=5, filters={"lang": "fr"})
    assert k["knn"]["num_candidates"] == 10 and k["knn"]["filter"] == [{"term": {"lang": "fr"}}]


def test_msearch_and_parse():
    body = build_msearch("cer-chunks", {"q": 1}, {"k": 2})
    assert body == [{"index": "cer-chunks"}, {"q": 1}, {"index": "cer-chunks"}, {"k": 2}]
    resp = {"hits": {"hits": [{"_id": "x", "_score": 3.5, "_source": {"chunk_id": "x", "doc_id": "D", "text": "t"}}]}}
    h = parse_hits(resp)[0]
    assert (h.chunk_id, h.doc_id, h.score) == ("x", "D", 3.5)


class FakeIndices:
    def __init__(self):
        self.calls = []

    def exists_alias(self, name):
        return True

    def get_alias(self, name):
        return {"cer-chunks-v1": {}}

    def update_aliases(self, actions):
        self.calls.append(actions)


class FakeES:
    def __init__(self):
        self.indices = FakeIndices()


def test_alias_swap_is_atomic():
    fake = FakeES()
    ElasticsearchBackend(BackendConfig(type="elasticsearch"), client=fake).swap_alias("cer-chunks-v2")
    assert fake.indices.calls == [[{"remove": {"index": "cer-chunks-v1", "alias": "cer-chunks"}},
                                   {"add": {"index": "cer-chunks-v2", "alias": "cer-chunks"}}]]


class FakeCreateIndices(FakeIndices):
    def __init__(self, existing):
        super().__init__()
        self.existing, self.deleted, self.created = set(existing), [], []

    def exists(self, index):
        return index in self.existing

    def delete(self, index):
        self.deleted.append(index)

    def create(self, index, settings, mappings):
        self.created.append(index)


def test_create_index_never_drops_the_served_index():
    fake = FakeES()
    fake.indices = FakeCreateIndices(existing=["cer-chunks-v1", "cer-chunks-v2"])  # alias -> v1
    be = ElasticsearchBackend(BackendConfig(type="elasticsearch"), client=fake)
    with pytest.raises(ValueError, match="index_version"):
        be.create_index(dim=8, version=1)
    assert be.create_index(dim=8, version=2) == "cer-chunks-v2"  # stale, not served: rebuilt
    assert fake.indices.deleted == ["cer-chunks-v2"] and fake.indices.created == ["cer-chunks-v2"]


class FakeSearchES(FakeES):
    def __init__(self):
        super().__init__()
        self.sent = None

    def msearch(self, searches):
        self.sent = searches
        hit = lambda d: {"hits": {"hits": [{"_id": d, "_score": 1.0, "_source": {"chunk_id": d, "doc_id": d}}]}}
        return {"responses": [hit("bm25doc"), hit("knndoc")][: len(searches) // 2]}


def test_hybrid_search_one_round_trip():
    fake = FakeSearchES()
    be = ElasticsearchBackend(BackendConfig(type="elasticsearch"), client=fake)
    b, k = be.hybrid_search("q", [0.1, 0.2], k_bm25=5, k_knn=5, num_candidates=50)
    assert len(fake.sent) == 4 and b[0].doc_id == "bm25doc" and k[0].doc_id == "knndoc"
    b, k = be.hybrid_search("q", None, k_bm25=5, k_knn=0, num_candidates=50)
    assert len(fake.sent) == 2 and k == []
