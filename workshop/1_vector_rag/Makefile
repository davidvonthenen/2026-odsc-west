PYTHON ?= python
INDEX ?= bbc-vector-chunks
DATA_DIR ?= ./bbc

.PHONY: help ingest query client slm-server

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

slm-server:
	$(PYTHON) slm_service.py
