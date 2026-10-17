DATASETS ?= nfcorpus scifact
.PHONY: setup up down test test-es download ingest eval report smoke all

setup:            ## install with embedding + dev extras
	pip install -e ".[embed,dev]"
up:               ## start Elasticsearch
	docker compose up -d elasticsearch && docker compose exec elasticsearch sh -c 'until curl -fs localhost:9200/_cluster/health?wait_for_status=yellow; do sleep 2; done'
down:
	docker compose down
test:             ## unit + e2e tests (no Elasticsearch, no download)
	pytest -q
test-es:          ## integration tests against the running Elasticsearch
	ES_URL=http://localhost:9200 pytest -q -m integration
download:
	for d in $(DATASETS); do cer download $$d; done
ingest:           ## one index per dataset (bm25/dense/hybrid share it)
	for d in $(DATASETS); do cer ingest configs/$${d}_hybrid.yaml; done
eval:
	for d in $(DATASETS); do cer eval configs/$${d}_bm25.yaml configs/$${d}_dense.yaml configs/$${d}_hybrid.yaml; done
report:
	cer report
smoke:            ## pipeline check without Elasticsearch or models
	cer download nfcorpus && cer eval configs/smoke_memory.yaml
all: up download ingest eval report
