PYTHON ?= python
INDEX ?= bbc-vector-chunks
DATA_DIR ?= ./bbc

.PHONY: help ner ner-server ner-client ingest query client slm-server reset-memory reset-mem0

help:
	@echo "Make sure you have the OpenSearch server running."
	@echo "Then run the following commands in this order:"
	@echo "1. make ingest"
	@echo "2. make query"

ingest:
	$(PYTHON) ingest.py

query:
	$(PYTHON) query.py
client: query

ner:
	$(PYTHON) ner_server.py
ner-server: ner

ner-client:
	$(PYTHON) ner_client.py

reset-memory:
	$(PYTHON) reset_memory.py
reset-mem0: reset-memory

slm-server:
	$(PYTHON) slm_service.py
