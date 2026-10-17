"""Runs only with a live Elasticsearch: ES_URL=http://localhost:9200 pytest -m integration"""
import os

import pytest

from cer.config import BackendConfig, RetrievalConfig
from cer.data import beir
from cer.embed import HashingEmbedder
from cer.index.es import ElasticsearchBackend
from cer.ingest.pipeline import ingest
from cer.retrieve.retriever import Retriever

pytestmark = pytest.mark.integration


@pytest.fixture
def backend():
    url = os.environ.get("ES_URL")
    if not url:
        pytest.skip("ES_URL not set")
    be = ElasticsearchBackend(BackendConfig(type="elasticsearch", url=url, alias="cer-test"))
    be.es.indices.delete(index="cer-test-v1,cer-test-v2", ignore_unavailable=True)  # leftovers of an aborted run
    yield be
    be.es.indices.delete(index="cer-test-v1,cer-test-v2", ignore_unavailable=True)


def test_ingest_search_and_swap(backend, mini_dir):
    emb = HashingEmbedder(64)
    rep = ingest(beir.iter_corpus(mini_dir), backend, emb, version=1, max_words=80, overlap_words=10,
                 batch_size=4, progress=False)
    assert backend.count() == rep.chunks
    r = Retriever(backend, emb, RetrievalConfig(mode="hybrid", k_bm25=10, k_knn=10, num_candidates=50))
    res = r.search("first line drug for type 2 diabetes")
    assert res.results[0].doc_id == "d1"
    rep2 = ingest(beir.iter_corpus(mini_dir), backend, emb, version=2, max_words=80, overlap_words=10,
                  batch_size=4, progress=False)
    assert list(backend.es.indices.get_alias(name="cer-test")) == [rep2.index_name]
    with pytest.raises(ValueError, match="index_version"):  # the served index is never rebuilt in place
        ingest(beir.iter_corpus(mini_dir), backend, emb, version=2, progress=False)
