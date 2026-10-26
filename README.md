# Open Data Science Conference (ODSC) AI West 2026

All resources (slides, code, etc) for ODSC West 2026: Cognitive Memory Architectures: Designing Continuity into Long-Running RAG and Agentic Soluitions

## Workshop Prerequisites

**IMPORTANT:** There is a super easy guide with a script to install and setup these various components ahead of this workshop. Please see [workshop/README.md](workshop/README.md).

To make the most of this interactive workshop, participants should ensure they have:

- A Linux or Mac-based developer laptop with enough memory (8GB+ minimum, 16GB recommended) to run a few podman/Docker containers.
  - Windows Users: The workshop should work, but untested. Recommend using a VM or Cloud Instance. This will NOT be provided.
- Python installed: 3.12 is recommended (but 3.10+ should work)
  - (Optional, but HIGHLY Recommended) Using a miniconda (recommended for beginners), uv, or venv virtual environment
  - Familiarity with Python is a must.
- Podman installed (recommended, as we will be using this) or Docker Desktop
  - Podman installation instructions: https://podman.io/docs/installation 
  - Pull container images BEFORE the day of the workshop:
    - podman pull docker.io/opensearchproject/opensearch:3.5.0
    - podman pull docker.io/opensearchproject/opensearch-dashboards:3.5.0
    - podman pull pgvector/pgvector:pg17
    - Build mem0 and mem0-dashboard using these instructions: https://github.com/mem0ai/mem0/blob/main/server/README.md
- Access to an OpenAI-compatible platform (REST API) with an API KEY
  - An API Key will NOT be provided
  - Compatible platforms: OpenAI, Nebius TokenFactory, Groq
  - Untested but should work: Together AI, Fireworks AI, or OpenRouter
- Create a FREE account on Tavily, an AI Search Engine (Using this for demo purposes)
  - Website: [https://www.tavily.com](https://www.tavily.com)

## Workshop Materials

To install prerequisites, [workshop/README.md](workshop/README.md).

Full materials coming soon!
