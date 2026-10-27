# Open Data Science Conference (ODSC) AI West 2026

All resources (slides, code, etc.) for **ODSC West 2026: Cognitive Memory Architectures: Designing Continuity into Long-Running RAG and Agentic Solutions**.

This hands-on workshop explores how to give Retrieval-Augmented Generation (RAG) applications continuity across interactions without confusing retrieved information with remembered experience. We start with a vector-only OpenSearch RAG pipeline and examine its limits using the evolving Windsurf acquisition story. We then add Real-Time web evidence and persistent memory, develop a domain-focused Agentic RAG News Agent, and finish by examining how compaction and expiration affect long-running memory.

The lab uses **Python**, **OpenSearch** for historical document retrieval, AI search for Real-Time web evidence, **mem0** for persistent memory, and **Model Context Protocol (MCP)** in the agentic exercises. The goal is to understand what each source contributes, how to trace evidence, and why remembering information is different from deciding whether it should still influence an answer.

| Section | High-level goal | Suggested time |
| --- | --- | --- |
| [1. Baseline RAG](workshop/1_vector_rag/README.md) | Build and query a vector-only document index; distinguish relevant evidence from complete, current, or accurate evidence; demonstrate the lack of conversational memory. | 15 minutes |
| [2. RAG + Real-Time Web Search + Memory](workshop/2_vector_rag_episodic/README.md) | Add Real-Time search results and mem0-based conversation and episodic recall; inspect sources and memory across separate Python invocations. | 20 minutes |
| [3. Agentic RAG + MCP + Memory](workshop/3_agentic_rag_episodic/README.md) | Run a domain-scoped News Agent that selects historical and current evidence, accesses Real-Time News through MCP, and preserves user context with mem0. | 20 minutes |
| [4. Long-Running Memory Lifecycle](workshop/4_agentic_memory_lifecycle/README.md) | Observe capture, retrieval, versioned conversation compaction, state updates, and expiration-based exclusion while retaining the underlying history. | 20 minutes |

## Workshop Prerequisites

**IMPORTANT:** There is a super easy guide with a script to install and setup these various components ahead of this workshop. Please see [workshop/0_setup/README.md](workshop/0_setup/README.md).

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

## Workshop Instruction

Follow the exercise READMEs in order. Each includes the commands to run, what to observe, and checkpoints for understanding the results.

1. **[Environment Setup](workshop/0_setup/README.md)**: Check host prerequisites, clone the repository, configure Python and API keys, and run the setup and validation scripts. The setup launches PostgreSQL/pgvector, mem0 and its dashboard, OpenSearch, and OpenSearch Dashboards. The README also covers starting, stopping, and cleaning up the workshop environment.

2. **[Exercise 1: Baseline RAG - Relevant Does Not Mean Correct](workshop/1_vector_rag/README.md)**: Ingest the supplied BBC news articles into an OpenSearch vector index and ask questions about the Windsurf transaction timeline. Inspect retrieved document chunks and compare what the corpus supports with the later Cognition announcement that is missing from the index. Test a user preference across separate requests to see why document retrieval does not provide interaction memory.

3. **[Exercise 2: RAG + Real-Time Web Search + Memory](workshop/2_vector_rag_episodic/README.md)**: Revisit the Windsurf questions with AI web search and mem0. Compare indexed document evidence, current web evidence, recalled memories, and restored conversation history using the `[DOC n]`, `[WEB n]`, `[MEM n]`, and `[TURN n]` markers. This exercise calls to a Real-Time AI search **directly through its Python client**, not through MCP.

4. **[Exercise 3: Agentic RAG + MCP + Memory](workshop/3_agentic_rag_episodic/README.md)**: Start the NER service, Real-Time search MCP endpoint, and Agentic News Agent. Investigate Google AI news while tracing agent routing, historical OpenSearch evidence (`[H#]`), current MCP evidence (`[W#]`), recalled mem0 context (`[M#]`), and audit records. Restart the agent to confirm that a prior user preference remains available.

5. **[Exercise 4: Long-Running Agentic RAG with Explicit Memory Lifecycle Policies](workshop/4_agentic_memory_lifecycle/README.md)**: Extend the same News Agent to inspect **Capture, Consolidate, Retrieve, Update, and Forget** events. Trigger message-count-based compaction, inspect versioned working-history snapshots, resume after a restart, and test time-based exclusion of expired memories. In this implementation, expiration removes records from ordinary topic recall; it does **not** physically delete the raw conversation or archived memory records.

The agentic exercises keep memory scoped to the **News Agent's domain**. Throughout Exercises 2-4, **mem0 remains the persistent memory service**; Exercise 4 adds lifecycle policies.

For a printable prerequisite overview, see [Cognitive Memory Architectures: Prerequisite Software](DavidvonThenen_Cognitive_Memory_Architectures_Prerequisite_Software.pdf).
