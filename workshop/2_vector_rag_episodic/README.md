# Exercise 2: RAG + Real-Time Web Search + Memory

**Cognitive Memory Architectures: Designing Continuity into Long-Running RAG and Agentic Solutions**  
**Section 2 of 4: Suggested time: 20 minutes**

In Exercise 1, you investigated Windsurf using a fixed document collection. Relevant articles were available, but the corpus was missing a later development. Each question also started without any record of your earlier interactions.

This exercise adds two capabilities: **current web search through Tavily** and **persistent memory through mem0**. You will repeat the Windsurf questions, inspect the evidence behind the answers, and demonstrate continuity across separate Python processes. Then you will start a new conversation with current web search disabled to see what earlier research can contribute.

## What you will learn

By the end of this exercise, you should be able to:

1. Inspect named entities and explain how entity labels support memory retrieval.
2. Compare an answer grounded in the indexed corpus with one that also uses web evidence.
3. Explain how the application can find the later Windsurf acquisition announcement without adding it to OpenSearch.
4. Trace claims through `[DOC n]`, `[WEB n]`, `[MEM n]`, and `[TURN n]` citation markers.
5. Distinguish restored conversation history from topic memories retrieved across sessions.
6. Demonstrate that remembered web evidence remains available when current web search is disabled.
7. Explain why citations, persistent storage, and a current search do not automatically establish that a claim is correct.

## Work from the exercise directory

Run every command from the Exercise 2 directory, `2_vector_rag_episodic`, containing `query.py`, `ner_server.py`, `ner_client.py`, and `common/`. Use the Python environment prepared during attendee setup. OpenSearch, mem0, Tavily credentials, and OpenAI configuration are already available. The `./bbc` dataset has already been ingested.

Open **two terminals** in that directory and activate the prepared Python environment in each one:

| Terminal | Use | Default service address or command |
| --- | --- | --- |
| A | NER server | [http://127.0.0.1:8000/ner](http://127.0.0.1:8000/ner) |
| B | NER client examples and all query commands | `python ner_client.py` and `python query.py` |

Leave the NER server running in Terminal A throughout this exercise. Run client requests one at a time in Terminal B and wait for each response before continuing. Each `python query.py` invocation answers one question and exits.

Keep the same configured `MEMORY_USER_ID` and `MEMORY_SESSION_ID` throughout Steps 1-5. Step 6 explicitly starts a new session for its memory experiment. The packaged defaults are `user-123` and `123456`; use the identities assigned during your setup if they differ. Existing history under those identities will be loaded, so your initial message counts may differ from another attendee's.

## The application you are using

`query.py` calls the application's web-search helper, which uses Tavily's Python client. OpenAI receives the current question together with several distinct kinds of context:

| Context | Where it comes from | What it contributes |
| --- | --- | --- |
| Indexed documents | OpenSearch vector-only retrieval | Relevant chunks from the workshop's fixed corpus |
| Current web results | A search performed for this question | Additional external evidence, including developments missing from the corpus |
| Recalled topic memories | mem0 search within the current user's memories | Earlier web-search episodes and model-extracted conversation memories, potentially from other sessions |
| Prior conversation | The current user/session's stored conversation | The ordered record of what you and the assistant previously said |

The sequence is programmed into the application. It restores the conversation, recalls relevant memories, searches the web, and saves newly encountered web episodes. It then retrieves document chunks, asks OpenAI for an answer, saves the updated conversation, and submits the latest question/answer pair to mem0 for memory extraction. These stages are visible in [query.py](query.py).

**A new search is performed for each question while web search is enabled.** Finding a relevant memory does not cause the application to skip that search. Also, a search performed now can return an older article: retrieval time and publication time are different.

## Start the named entity recognition server

**Terminal A**

**Named entity recognition (NER)** identifies spans of text representing entities such as organizations, people, places, and products, and assigns category labels. For example, an extractor might label `Google` as `ORG` and `California` as `GPE`, a geopolitical entity. These are predictions, so the output depends on the text and model. See the [spaCy NER explanation](https://spacy.io/usage/linguistic-features#named-entities).

This service uses spaCy's `en_core_web_sm` pipeline by default. If the pipeline was not installed during attendee setup, install it once in your prepared Python environment using [spaCy's download command](https://spacy.io/usage/models#download):

```bash
python -m spacy download en_core_web_sm
```

Start the server:

```bash
python ner_server.py
```

Wait for the server to report that it is listening on port `8000`, then leave it running.

**This application's memory workflow depends on the NER service.** The conversation manager calls it to extract entities from questions and to annotate conversation records and web-search episodes before saving them to mem0. It combines each entity's label and normalized name into values such as `ORG:google`, which help select memories about the same entities during later retrieval.

NER adds topic metadata; it does not establish whether a news claim is true.

Implementation: [NER server](ner_server.py) and [entity labeling in the conversation manager](common/conversation.py).

## Try the NER client

**Terminal B**

**Reminder:** Activate the same prepared Python environment in this terminal, whether you use `venv`, Miniconda, `uv`, or another environment manager.

Send a short sample sentence to the server:

```bash
python ner_client.py --text "Google and Microsoft are mentioned in a report from California."
```

Inspect these response fields:

| Field | What to look for |
| --- | --- |
| `model` | The spaCy pipeline used for extraction; `en_core_web_sm` by default |
| `entities` | Deduplicated, lowercase entity names |
| `entity_pairs` | Extracted names and their category labels, in `name` and `label` fields |
| `request_id` | An identifier for this NER request |

A typical result may include `google` and `microsoft` with `ORG`, and `california` with `GPE`. Inspect your output rather than requiring an exact match to that example.

Now try a sentence containing a person and places:

```bash
python ner_client.py --text "My name is David and I am from Long Beach, California."
```

Which words became entities? Which did not? Did the extractor recognize a place containing more than one word?

Look at Terminal A for the `/ner called` and `[entities discovered]` messages. Compare the logged names with the JSON response in Terminal B.

These two client commands inspect entity extraction. They do not save the sample sentences to mem0. Memory capture happens as part of the `query.py` requests below.

During topic-memory retrieval, the application extracts entities from the question and obtains semantic candidates from mem0. If candidates share an entity label and name with the question, it selects from that matching subset in similarity order. If the question has no extracted entities or none of the candidates share them, it uses the semantic candidates. NER therefore supplements similarity search, with a fallback when entity overlap is absent.

Implementation: [NER client](ner_client.py) and `retrieve_topic_memories()` in the [conversation manager](common/conversation.py).

**Continue in Terminal B for all remaining commands. Keep the NER server running in Terminal A.**

## Step 1: Repeat the OpenAI question

Run the first question from Exercise 1:

```bash
python query.py --question "How much did OpenAI purchase Windsurf for?"
```

An example observed response to examine is:

```text
ANSWER: OpenAI purchased Windsurf for about **$3 billion**. [DOC 1]
```

**Treat that wording as a claim to critique.** The earlier report described an agreement that had not closed. Saying "purchased" turns the reported transaction into a completed event. Your run may already correct the question's premise; record what it actually says.

**Observe:**

- How does the answer compare with your Exercise 1 answer?
- Does it distinguish a reported agreement from a completed acquisition?
- Which entries appear under `RAG Hits` and `Internet Hits`?
- Does the answer cite document evidence, web evidence, or both?
- Does the footer show any `New internet episodes saved`?

**What changed?** The application can now obtain evidence beyond the corpus and preserve information from this interaction. That additional context gives it more material to evaluate, but does not guarantee that it will challenge every misleading premise.

If this is the first request for a previously unused user, `Recalled Episodic Memories` can be empty even when new web episodes are saved. Recall happens **before** this request's web results are stored. Those results become candidates for subsequent requests.

## Step 2: Repeat the Google question

Run:

```bash
python query.py --question "How much did Google purchase Windsurf for?"
```

An example observed response is:

```text
ANSWER: Google didn't purchase Windsurf itself. Based on the evidence, Google paid about **$2.4 billion** for Windsurf's technology licensing, talent, and compensation, while **OpenAI** had earlier been reported to be in talks to buy Windsurf for about **$3 billion** [DOC 1][DOC 2][WEB 3].
```

The wording and marker numbers may differ. Check the cited sources for the transaction's exact terms; the Reuters report describes nonexclusive licensing fees alongside key hires.

**Compare with Exercise 1:**

- Does the answer distinguish licensing and hiring from purchasing the company?
- What does the web evidence add to the document evidence?
- Can you identify the current result associated with `[WEB 3]`, if that marker appears in your answer?

Also look near the beginning of the output for a log such as:

```text
Loaded 2 messages from Mem0 run '123456-complete'
```

For a fresh session, the earlier question and answer account for those two messages. A different count is expected if the session already contains history. **The previous Python process has exited, but its conversation has been restored from persistent storage.**

The answer's exact contents depend on retrieved evidence, saved history, and generation. Compare the evidence and distinctions, rather than treating an example sentence as a required output.

## Step 3: Ask who ultimately acquired Windsurf

Run:

```bash
python query.py --question "Which company ultimately acquired Windsurf?"
```

The target conclusion is **Cognition**, supported by relevant external evidence. An example response is:

```text
ANSWER: Windsurf was ultimately acquired by **Cognition**. [WEB 1][WEB 2][WEB 4][WEB 5]
```

**Inspect the evidence:**

1. Find a relevant acquisition announcement or subsequent report in `Internet Hits`.
2. Open its URL and check the company, date, and transaction language.
3. Match the answer's web markers to the results supplied in this request.
4. Compare the evidence with the `RAG Hits`. Which part of the story was missing from the indexed collection?

### Reconstruct the timeline

Use these sources to distinguish the stages of the story:

| Date | Source | What the source establishes | In the workshop corpus? |
| --- | --- | --- | --- |
| May 6, 2025 | [Reuters: Reported OpenAI agreement](https://www.reuters.com/business/openai-agrees-buy-windsurf-about-3-billion-bloomberg-news-reports-2025-05-06/) | A roughly $3 billion agreement was reported; it had not closed. | Yes |
| July 11-12, 2025 | [Reuters: Google's Windsurf arrangement](https://www.reuters.com/business/google-hires-windsurf-ceo-researchers-advance-ai-ambitions-2025-07-11/) | Approximately $2.4 billion in nonexclusive licensing fees, with key personnel joining Google; Google took no stake or controlling interest. | Yes |
| July 14, 2025 | [Cognition: Acquisition announcement](https://cognition.com/blog/windsurf) | Cognition announced a signed definitive agreement to acquire Windsurf, including its product and intellectual property. | No |

The two indexed articles describe different developments in a timeline. Apparent contradictions arise when an answer collapses those developments into interchangeable claims about who bought the company. The OpenAI transaction did not become the eventual acquisition; Cognition became the acquirer. Keep the distinction between an agreement announcement and evidence of completion when reading any individual source. [Windsurf's announcement](https://devin.ai/blog/windsurfs-next-chapter) also identifies Cognition as the acquiring company.

**Why Exercise 2 can answer differently:** Tavily can retrieve evidence missing from the fixed corpus. The application supplies the retrieved snippets to OpenAI and saves new web episodes in mem0. It does not add those pages to the OpenSearch document index.

If your answer differs, inspect the returned URLs and what they support. Search rankings, page availability, and generated wording vary. A correct company name alone does not establish that this request retrieved supporting evidence. To focus a follow-up search on the missing development, you can try:

```bash
python query.py --question "What does Cognition's July 14, 2025 announcement say about acquiring Windsurf?"
```

## Step 4: Read the citations and memory diagnostics

### Follow a claim back to its context

The application supplies labeled evidence to OpenAI and asks it to cite those labels. The terminal lists the corresponding items in the same order, although the lists themselves do not print the bracketed labels.

| Marker | Meaning in this request | How to inspect it |
| --- | --- | --- |
| `[DOC 1]` | First retrieved OpenSearch chunk | First item in `RAG Hits`; inspect its source document when checking the claim |
| `[WEB 3]` | Third result from this request's web search | Third item in `Internet Hits`, including its title and URL |
| `[MEM 2]` | Second recalled topic memory | Second item in `Recalled Episodic Memories` |
| `[TURN 1]` | First message in the restored prior conversation | Your earlier conversation; the full numbered transcript is included in the prompt, but is not printed as a terminal section |

These numbers are assigned for the current request. `[WEB 1]` in an earlier answer may refer to a different page from `[WEB 1]` now. Old citation strings embedded in stored answers do not establish a source mapping for a new answer.

**Try it:** Choose one claim in the acquisition answer, follow its marker to the corresponding source, and read enough to determine whether the source supports the precise wording. Then check its date. A citation helps you inspect a claim; the code does not independently verify that every generated claim follows from its citation.

`RAG Hits` counts document **chunks**, so several entries can come from the same article. Similarity scores indicate retrieval relevance, not factual correctness. Recalled memory previews are shortened to 120 characters, so the terminal preview may omit the supporting detail supplied to the model.

### Understand the footer

Your run will print these fields:

| Field | What it tells you |
| --- | --- |
| `Docs provided to LLM` | Number of retrieved document chunks supplied to OpenAI; the default requested maximum is 3 |
| `Current web search enabled` | Whether current web retrieval was enabled; `True` alone does not establish that results were returned |
| `Internet results provided to LLM` | Number of current web results supplied; the default requested maximum is 5 |
| `New internet episodes saved` | Number of new web-result episodes saved during this invocation, after duplicate checks |
| `Episodic memories recalled` | Number of previously stored topic memories selected for this request; default maximum 5 |
| `Conversation chunks saved` | Fragments written for the complete updated conversation snapshot, including earlier messages |
| `Synthesized turn writes` | Number of question/answer submissions for mem0 inference, usually 1; it is not the number of facts extracted |
| `Conversation messages persisted` | Total user and assistant messages in this session after adding the current question and answer |

For a previously empty session, three successful questions normally leave **six conversation messages**. Long messages can span multiple storage chunks. Each new snapshot includes the full conversation, and earlier snapshots remain stored.

A repeated question can retrieve current web results while saving fewer or no new web episodes. Within a session, the duplicate check uses the exact question and result URL. Rewording a question or receiving different URLs can produce new episodes. The duplicate path does not refresh the archived payload when the same question and URL reappear.

### Recognize the three stored content types

The implementation keeps these distinct in mem0:

| Content | Run suffix and memory kind | How it is used later |
| --- | --- | --- |
| Complete conversation | `-complete`; `conversation_chain` | Reconstructs the current session's ordered messages without summarizing them |
| Inferred conversation memories | `-synthesized`; `synthesized_conversation` | Supplies model-extracted context from earlier question/answer pairs |
| Archived web results | `-internet`; `internet_search_episode` | Preserves the earlier search question, provider, title, URL, returned content, and score |

The raw conversation and web records use `infer=False`. The question/answer pair uses `infer=True`, allowing mem0 to extract memories. One inference submission can yield no extracted memory, one memory, or several; the printed write counter does not measure that result.

**The distinction matters:** a stored search result records what an earlier search returned. An inferred memory may contain assistant-generated claims. A prior conversation records what someone said. Each has provenance, but none becomes verified world knowledge merely by being stored. The source-formatting functions in [common/common.py](common/common.py) preserve those distinctions in the prompt.

## Step 5: Repeat the preference test from Exercise 1

State the same preference used in the baseline exercise:

```bash
python query.py --question "For future answers, I prefer short bullet points. What was OpenAI's reported Windsurf deal?"
```

Wait for the command to finish. Then start another invocation:

```bash
python query.py --question "What response format did I tell you I prefer in my previous question?"
```

**Look for:**

- A startup log reporting that previous conversation messages were loaded.
- An answer identifying your preference for **short bullet points**.
- A citation such as `[TURN n]` pointing to the earlier user statement.
- An increased `Conversation messages persisted` count.

Unlike Exercise 1, the second process has access to the earlier interaction through mem0. Your previous preference is present in the restored conversation even if mem0's topic search does not select a separate preference memory.

**What did you demonstrate?** Durable conversation continuity across process exits. This test alone does not establish that a preference was extracted into a distinct semantic memory: full transcript restoration is sufficient to answer it. The next step separates that transcript path from topic recall.

## How the implementation works

Use this map if you want to follow the behavior you observed. No code changes are required.

| File | What to look at |
| --- | --- |
| [query.py](query.py) | One-question CLI; loading history; retrieving memories; web search; writes; displayed evidence and counters |
| [common/web_search.py](common/web_search.py) | The web-search switch and provider selection; Tavily is the default, with DuckDuckGo as the configured default fallback |
| [common/tavily_client.py](common/tavily_client.py) | Sending the exact question to Tavily and normalizing returned titles, URLs, snippets, and scores |
| [common/common.py](common/common.py) | Vector retrieval, the four labeled context blocks, prompt construction, and the OpenAI request |
| [common/conversation.py](common/conversation.py) | Conversation snapshots, inferred turn writes, raw internet episodes, duplicate checks, and topic recall |
| [common/mem0_client.py](common/mem0_client.py) | Remote memory operations and user/run scoping |
| [common/config.py](common/config.py) | Identity settings, retrieval limits, and `WEB_SEARCH_ENABLED` |
| [ner_server.py](ner_server.py) | The spaCy NER service, normalized entity names, and category labels returned by `/ner` |
| [ner_client.py](ner_client.py) | The NER demonstration CLI and HTTP client used by the conversation manager |

If a fallback provider appears under `Internet Hits`, use that displayed provider when interpreting the run. It identifies where those results actually came from.

The answer policy in `SOURCE_AUTHORITY_POLICY` is **claim-specific**. An explicit user statement supports a user preference. A relevant external source supports a world fact. Source dates and provenance matter when evidence conflicts. The policy does not assign automatic priority to every web result over every indexed document or archived source. It guides generation; it does not rewrite conflicting stored memories.

## Finish: What did this exercise demonstrate?

Before moving on, confirm that you can explain these observations:

- [ ] I started the NER server and inspected entity names and labels with both client examples.
- [ ] I can explain how entity overlap influences topic-memory selection and when semantic candidates are used instead.
- [ ] I repeated the OpenAI and Google questions and compared both answers with Exercise 1.
- [ ] I distinguished a reported agreement, a licensing/hiring arrangement, and the eventual acquisition.
- [ ] I asked who ultimately acquired Windsurf and inspected the evidence supporting the answer.
- [ ] I traced a claim through its current-request citation marker to a source.
- [ ] I observed web episodes being saved and relevant earlier memories being recalled.
- [ ] I demonstrated that my stated preference was available in a subsequent Python process.
- [ ] I tested a new session with current web search disabled and inspected the role of cross-session memory.
- [ ] I can distinguish full conversation restoration, inferred memories, and archived search evidence.

### Carry this result into the next exercise

You now have a fixed RAG workflow with access to additional web evidence and durable interaction history. These capabilities address different needs: web retrieval can fill gaps in the corpus, while memory can preserve the context of earlier work.

The next exercise introduces decisions about which context sources a request needs. Later, the workshop examines explicit policies for consolidating useful memories, updating outdated conclusions, and forgetting information. This implementation already captures and retrieves memory, but does not automatically compact conversation history or apply expiration and forgetting policies. Its growing record gives you something concrete to govern.

Move onto the next [Exercise 3: Agentic RAG + MCP + Memory](../3_agentic_rag_episodic/README.md).
