# Open Data Science Conference (ODSC) AI West 2026

All resources (slides, code, etc) for ODSC West 2026: Cognitive Memory Architectures: Designing Continuity into Long-Running RAG and Agentic Soluitions

## Create an Account on Tavily

Create a FREE account on Tavily, an AI Search Engine (Using this for demo purposes).

Website: [https://www.tavily.com](https://www.tavily.com)

Set the API Key in your environment variables as `TAVILY_API_KEY`.

## Podman Setup

Either `podman` (Recommended) or `docker` is required. If you use `docker`, then you need to change all the commands within the scripts from  `podman` -> `docker` OR create an alias.

Install Podman using the [official Podman instructions](https://podman.io/docs/installation).

Once `podman` is installed, to setup/build the containers required in this session, run:

```bash
$ ./setup.sh
```

To stop the newly built containers, run the following:

```bash
$ ./stop.sh
```

## Python Software Prerequisites

It's recommended you are using a virtual environment like `miniconda`, `uv`, or `venv`.

To install the software dependencies for Python:

```bash
pip install -r requirements.txt
```

## Validate the Environment

To validate your environment ahead of the workshop, run the following:

```bash
python validate_environment.py
```

Iterate until this says "Successful"!

## Workshop Materials

Coming soon!
