# Exercise 1: Baseline RAG - Relevant Does Not Mean Correct

**Cognitive Memory Architectures: Designing Continuity into Long-Running RAG and Agentic Solutions**  
**Section 1 of 4: Suggested time: 15 minutes**

In this exercise, you will use a basic Retrieval-Augmented Generation (RAG) application to investigate a real sequence of news stories about Windsurf. The application retrieves relevant documents from OpenSearch and sends them, with your question, to OpenAI to generate an answer.

It is a useful starting point, but it has two intentional limitations: its evidence comes only from a fixed document corpus, and it remembers nothing from previous interactions. You will observe both limitations before adding more capabilities in subsequent exercises.

## What you will learn

By the end of this exercise, you should be able to:

1. Verify the workshop's previously configured environment by successfully ingesting documents and answering questions.
2. Explain the path from document chunking and vector embeddings to an OpenSearch nearest-neighbor search and a generated answer.
3. Recognize that a semantically relevant document is not necessarily current, complete, or sufficient to verify a factual claim.
4. Explain why independent evidence, such as a second source or a current-information retrieval tool, is often needed to validate a result.
5. Demonstrate that this baseline does not remember prior user interactions, giving us a reference point for the memory-enabled exercises.

## The application you are using

```text
Question
   |
   v
Generate question embedding
   |
   v
OpenSearch vector-only k-NN search
   |
   v
Top-k document chunks + source metadata
   |
   v
Build grounded prompt (current question + retrieved chunks)
   |
   v
OpenAI
   |
   v
Answer + RAG hit metadata
```

The source documents and their embeddings persist in OpenSearch. **Your questions, answers, preferences, and conversation history do not.** Each run executes the same fixed sequence; there is no agent deciding when to retrieve, no memory retrieval, and no live-news lookup.

We will work from the project directory containing `ingest.py`, `query.py`, `common/`, and the provided `bbc/` dataset. Python, OpenSearch, and OpenAI were configured in the preceding setup section, so no additional setup or service checks are needed here.

## Step 1: Ingest the document corpus

Run:

```bash
python ingest.py
```

**What is happening?** The ingestion pipeline reads a dataset of text file from the British Broadcasting Corporation found at `./bbc`, splits each document into overlapping chunks, generates an embedding for each chunk, and stores the chunk plus source metadata in an OpenSearch vector index. The code uses the configured sentence-transformers embedding model, a `knn_vector` field, and an index named `bbc-vector-chunks` by default.

The default chunk settings in `ingest.py` are **2,048 characters per chunk** with a **256-character overlap**. Embeddings are generated in batches of 32. The stored metadata includes the source path, title, and category. The first run creates the index if it does not exist; subsequent runs replace chunks for each ingested source path.

**Observe:** The `Indexing` progress display and the final log message reporting how many chunks were ingested. A successful ingestion is our practical confirmation that the earlier Python/OpenSearch setup is usable.

**Think about it:** OpenSearch now has document knowledge. Has the application learned anything about you, your previous questions, or which news article is the most recent? **No.** An embedding represents text for similarity search; it is not a record of what the application has experienced.

## Step 2: Ask what OpenAI paid for Windsurf

Run the first required question:

```bash
python query.py --question "How much did OpenAI purchase Windsurf for?"
```

You will see output organized around these fields (the actual answer, sources, scores, and elapsed time will vary):

```text
QUESTION: How much did OpenAI purchase Windsurf for?

ANSWER: <generated response>

RAG Hits:
  - <document title> | <category> | score=<similarity score>

Query time: <elapsed time>s   (OpenSearch vector k-NN + LLM service generation)
Docs provided to LLM: <number of retrieved chunks>
```

**Record your observations:**

- What amount, if any, does the answer report?
- Does it describe a **proposed/reported deal** or a **completed purchase**?
- Which document titles appear under `RAG Hits`?
- How many document chunks were supplied to the model?

The original May 2025 Reuters story reported a possible OpenAI acquisition for approximately **$3 billion** and explicitly stated that the deal **had not yet closed**. Notice that the question itself assumes a purchase took place. A sound answer must not turn a reported agreement into proof of a completed acquisition.

**Important:** `RAG Hits` displays compact source metadata and similarity scores, not a fact-verification score. By default the application retrieves **three chunks**, not necessarily three different articles. A high similarity score means a chunk is relevant to the question; it does not establish that its claim is true, up to date, or complete.

## Step 3: Ask what Google paid for Windsurf

Run the second required question:

```bash
python query.py --question "How much did Google purchase Windsurf for?"
```

**Record your observations:**

- What amount appears in the generated answer?
- Does the answer say Google **acquired Windsurf**, or that Google **licensed technology and hired employees**?
- Are the `RAG Hits` different from those in Step 2?
- Did the application distinguish the date and type of the Google arrangement from the earlier OpenAI report?

The July 2025 Reuters story reported that Google would pay approximately **$2.4 billion in nonexclusive licensing fees** and hire key Windsurf personnel. Google did **not** acquire a controlling interest in Windsurf. That is not the same kind of transaction as a purchase of the company.

**Compare your two answers.** Both questions include a potentially misleading premise: that a company *purchased* Windsurf. The retrieved articles discuss large dollar amounts and overlapping parties, but the amount alone does not tell you what happened. These are developments in a timeline, not necessarily two interchangeable or equally current answers to the same question.

## Step 4: Reconstruct the full timeline

Open and read these three original sources in chronological order:

| Date | Source | What it establishes | Available in our RAG corpus? |
| --- | --- | --- | --- |
| May 6, 2025 | [Reuters: Reported OpenAI deal](https://www.reuters.com/business/openai-agrees-buy-windsurf-about-3-billion-bloomberg-news-reports-2025-05-06/) | A roughly $3 billion OpenAI acquisition was reported; the deal had **not closed**. | **Yes** |
| July 11, 2025 | [Reuters: Google's Windsurf arrangement](https://www.reuters.com/business/google-hires-windsurf-ceo-researchers-advance-ai-ambitions-2025-07-11/) | Google arranged roughly $2.4 billion in nonexclusive licensing fees and hired key personnel; it did **not** purchase Windsurf. | **Yes** |
| July 14, 2025 | [Cognition: Acquisition announcement](https://cognition.com/blog/windsurf) | Cognition announced that it had signed a definitive agreement to acquire Windsurf, including its product and intellectual property. | **No** |

**The key discovery:** The corpus includes the first two developments but **not the third**. It is missing the later primary-source announcement from Cognition. Reading only the indexed articles does not give the application the complete story of what ultimately happened to Windsurf.

Keep this distinction in mind:

- **Retrieval relevance:** Did the vector search find chunks about the requested subject?
- **Factual support:** Do those chunks substantiate the exact claim in the generated answer?
- **Temporal completeness:** Do the available sources cover subsequent developments that could change the answer?

Even perfect vector search cannot retrieve an article that was never ingested. The current system prompt explicitly directs the model to prioritize retrieved context, even when that differs from its existing knowledge about later events. This is deliberate for the baseline: we can inspect what the **corpus** supports rather than assume the model knows the current story.

## Step 5: Challenge the baseline with a later-event question

Now ask a question that depends on the missing article:

```bash
python query.py --question "Which company ultimately acquired Windsurf?"
```

Compare the answer with Cognition's announcement in Step 4.

**What to look for:** The application may answer using incomplete earlier coverage, hedge, abstain, or provide a correct name without supporting it from the retrieved documents. Do not treat any one of these behaviors as guaranteed. The point is to inspect the actual `RAG Hits` and ask whether **the evidence provided to the model** substantiates its conclusion.

An answer that happens to be correct is not automatically a **grounded** answer. If the necessary evidence is absent from the retrieved chunks, the application cannot demonstrate where that conclusion came from. Likewise, a confident answer based on old reporting is not proof that an acquisition was completed.

**Discussion:** Would increasing the number of nearest-neighbor results fix this problem? It might expose an overlooked *indexed* source, but it cannot recover the missing July 14 announcement. To verify developing events reliably, the application needs access to relevant, independently checked, and sufficiently current evidence.

## Step 6: Test whether the application remembers you

We have tested the application's document recall. Now test its **interaction recall**.

First, state a preference while asking a normal research question:

```bash
python query.py --question "For future answers, I prefer short bullet points. What was OpenAI's reported Windsurf deal?"
```

Then start a new invocation:

```bash
python query.py --question "What response format did I tell you I prefer in my previous question?"
```

**Observe:** Does the second answer actually demonstrate knowledge of the earlier preference? Check the retrieved documents. The first question and preference were not stored, and the second request contains no previous conversation messages. Any claim to remember your preference is unsupported by this application's data flow; the application may instead say it does not know.

This does not require deleting an index or restarting any service. Each `python query.py` command starts an independent process and passes only the current question and retrieved document snippets to OpenAI.

The distinction is fundamental:

> **RAG retrieves information about the world from a document collection. Memory preserves useful information about what happened in previous interactions.**

## Optional experiment: Does retrieving more chunks help?

The default configuration in `common/config.py` uses `RAG_TOP_K=3`. Change `RAG_TOP_K` to `5` in the configuration established during setup, and rerun:

```bash
python query.py --question "Which company ultimately acquired Windsurf?"
```

Compare the answer, returned hit titles, and `Docs provided to LLM` with your earlier run. Additional chunks may help if the relevant evidence is already indexed, but can also introduce distracting or conflicting material. More chunks are not necessarily better evidence.

Restore `RAG_TOP_K=3` after the experiment so the remaining workshop exercises begin with the same retrieval configuration.

## How the implementation works

You do not need to modify the code. Use this map to follow the path of a question through the actual application:

| File | What to look at |
| --- | --- |
| `ingest.py` | Reading `./bbc`, chunking, generating embeddings, and indexing source metadata |
| `common/config.py` | The OpenSearch index, embedding model, `RAG_TOP_K`, and OpenAI configuration |
| `common/embeddings.py` | Encoding text into normalized embedding vectors |
| `common/opensearch_client.py` | The `knn_vector` mapping and vector-only k-NN search |
| `common/common.py` | Converting the question into an embedding, formatting retrieved context, assembling the two-message prompt, and requesting a completion |
| `query.py` | The `--question` argument and the displayed answer, hit metadata, and timing |

The default retrieval path has no keyword search, reranking, secondary verification, current-news tool, or memory read/write step. That absence is intentional, not an exercise to fix by adding every feature at once.

## Finish: What did this baseline demonstrate?

Use this checklist before moving on:

- [ ] `python ingest.py` successfully ingested the provided corpus into OpenSearch.
- [ ] Both required Windsurf questions returned answers and identifiable `RAG Hits`.
- [ ] I can distinguish the reported **$3 billion OpenAI acquisition** from a completed purchase.
- [ ] I can distinguish Google's **$2.4 billion licensing/hiring arrangement** from an acquisition of Windsurf.
- [ ] I identified the July 14 Cognition announcement as evidence **missing from the indexed corpus**.
- [ ] I understand why semantic similarity does not establish factual accuracy, source reliability, or recency.
- [ ] I demonstrated that a preference stated in one command does not reliably carry over to the next.

### Carry this result into the next exercise

We now have a baseline: a functioning application that can retrieve relevant document chunks but cannot independently establish the latest external facts or remember interactions across requests.

In **Exercise 2: RAG + MCP + Memory**, we will keep the vector retrieval foundation and introduce two different capabilities:

- **Current external evidence through MCP**, so information can be checked beyond the frozen corpus.
- **Persistent memory**, so meaningful preferences, decisions, and events can influence later interactions.

These solve different problems. **Memory cannot make an outdated article current, and access to current information does not make an application remember its user.** Keeping those boundaries clear is the foundation for the governed, traceable, and auditable agent we will build throughout the workshop.

Move onto the next [Exercise 2: RAG + Memory](../2_vector_rag_episodic/README.md).
