PYTHON ?= python
QUESTION ?= What is Google currently working on in Artificial Intelligence (AI)?

.PHONY: install ingest mcp mcp-server ner ner-server ner-client server agent client query slm-server check clean

install:
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) -m src.ingest

ner:
	$(PYTHON) ner_server.py
ner-server: ner

ner-client:
	$(PYTHON) -m src.ner_client

mcp:
	$(PYTHON) -m src.tavily_mcp_server
mcp-server: mcp

server: agent
agent:
	$(PYTHON) -m src.news_agent

client: query
query:
	$(PYTHON) -m src.query --question "$(QUESTION)"

slm-server:
	$(PYTHON) slm_service.py

check:
	$(PYTHON) -m compileall -q src tests client.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
