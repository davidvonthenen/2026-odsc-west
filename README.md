# Open Data Science Conference (ODSC) AI West 2026

All resources (slides, code, etc) for ODSC West 2026: Cognitive Memory Architectures: Designing Continuity into Long-Running RAG and Agentic Soluitions

## Workshop Prerequisites

To make the most of this interactive workshop, participants should ensure they have:

- A Linux or Mac-based developer laptop with enough memory (8GB+ minimum, 16GB recommended) to run a few podman/Docker containers.
  - Windows Users should use a VM or Cloud Instance. This will NOT be provided.
- Python installed: 3.12 is recommended (the only version I tested with and can guarantee will work)
  - (Optional, but HIGHLY Recommended) Using a miniconda (recommended for beginners), uv, or venv virtual environment
- Familiarity with Python is a must.
- Podman installed (recommended, as we will be using this) or Docker Desktop
  - Podman installation instructions: https://podman.io/docs/installation 
  - Pull container images BEFORE the day of the workshop:
    - Build Script (in this folder which automates the Manual Process below): [setup.sh](setup.sh)
    - Manual method: pre-pull the following Podman images:
      - podman pull docker.io/opensearchproject/opensearch:3.5.0
      - podman pull docker.io/opensearchproject/opensearch-dashboards:3.5.0
      - podman pull pgvector/pgvector:pg17
      - Build mem0 and mem0-dashboard using these instructions: https://github.com/mem0ai/mem0/blob/main/server/README.md
- Access to an OpenAI-compatible platform (REST API) with an API KEY
  - An API Key will NOT be provided
  - Compatible platforms: OpenAI, Nebius TokenFactory, Groq
  - Untested but should work: Together AI, Fireworks AI, or OpenRouter
  - If you don't have access to an API Key, you can use a local Small Language Model, but that requires downloading these models and dropping them in ~/models on your laptop. (Requires a recommended 16GB of memory)
    - Apple laptops: https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-1M-4bit
    - All other CPUs: https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/blob/main/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf 


## Workshop Materials

To install prerequisites, [workshop/README.md](workshop/README.md).

Full materials coming soon!
