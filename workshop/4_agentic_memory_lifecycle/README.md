# Exercise 4: Long-Running Agentic RAG with Explicit Memory Lifecycle Policies

**Workshop:** Cognitive Memory Architectures: Designing Continuity into Long-Running RAG and Agentic Solutions  
**Section 4 of 4: Suggested time: 20 minutes**

Exercise 3 gave the News Agent continuity across interactions. This exercise adds policies for controlling the conversation it carries forward and the memories it can recall. You will investigate Anthropic, watch the agent compact an accumulated conversation, recover that compacted state after a restart, and observe temporary memories become ineligible for retrieval.

The architecture remains one News Agent using OpenSearch for historical news, Tavily through MCP for current news, and **mem0 for agent memory**. The lifecycle rules live in the Python application. Persistent storage remembers what happened; these rules decide how that history participates in later work.

## What you will learn

- Distinguish the raw conversation archive, active working history, inferred conversation memories, and archived web episodes.
- Explain how **compaction** controls repeated conversation context and how **expiration** controls memory eligibility.
- Observe Capture, Consolidate, Retrieve, Update, and Forget through console events and stored metadata.
- Verify continuity and expiration using history versions, record IDs, and deadlines.

## What changed from Exercise 3?

| Aspect | Exercise 3: Agentic RAG + MCP + Memory | Exercise 4: Explicit memory lifecycle |
| --- | --- | --- |
| Retrieval and services | One News Agent, OpenSearch, Tavily MCP, NER, and mem0 | The same components |
| Conversation capture | Raw role-labeled conversation, inferred Q&A memories, and archived web episodes | Those paths continue, with additional versioned working-history snapshots |
| Active conversation | Raw history and recalled context support continuity | The latest committed working history is restored before routing the next question |
| Compaction | No automatic working-message compaction | Completed user/assistant messages trigger a continuity summary at a configurable threshold |
| Expiration | Topic memories have no application expiration deadline | Inferred Q&A memories and archived web episodes receive UTC deadlines and are filtered when expired |
| Recovery and provenance | Raw conversation checkpoints and request audits | Immutable working snapshots, compaction versions, parent links, content hashes, and audit references |
| Visibility | Request and memory audit information | Lifecycle events plus a final five-operation summary for each chat request |

NER labeling, domain/user scoping, Q&A inference, source provenance, and web-result capture already existed in Exercise 3. The main additions here are **versioned compaction, expiration, and lifecycle observability**.

Implementation references: [memory configuration](src/common/config.py), [conversation capture and expiration](src/common/conversation.py), [working-history storage](src/common/message_history.py), and [agent lifecycle orchestration](src/news_agent/news_agent.py).

## Understand the two lifecycle controls

### Compaction: reduce the active conversation

The context window must accommodate the current question, instructions, evidence, and conversation history. Repeatedly carrying a growing transcript increases the text supplied to later model calls. Compaction replaces the active user/assistant history with one continuity summary while preserving the underlying archive.

The application first saves the completed conversation snapshot. When its message count reaches `compact_k` (Default value is just for demonstration purposes. For your specific application, benchmark a middle ground based on your context window.), it asks the generator to summarize topics, user statements, corrections, decisions, relevant dates, and unresolved questions. It stores that summary as a new version. The next turn restores the summary and adds the new exchange.

This trades detailed active context for a shorter representation. It can reduce repeated input tokens on later turns, but producing the summary adds a model call and can lose nuance. The trigger counts **messages**, not tokens; a few long messages can still consume substantial context. The summary body is limited to 4,000 characters before its continuity label is added. This endpoint reports `usage: null`, so the exercise demonstrates context management rather than measured token-cost savings.

### Expiration: limit how long a memory can be recalled

Each new inferred Q&A submission and archived web episode receives an absolute UTC `expiration` timestamp. The duration comes from `memory_expiration`, measured in **minutes**. Once the current time is later than that deadline, the record is excluded from the topic-memory retrieval path. Reading a record does not renew its deadline.

This is useful for changing information: an older observation should not retain unlimited eligibility merely because it is semantically similar to the next question. Expiration can reduce stale context competing for attention and prompt space. It does not establish whether an article is true or current; source quality and publication dates still matter.

| Stored information | Subject to this expiration policy? | Effect of compaction or expiration |
| --- | --- | --- |
| Raw user/assistant conversation | No | Retained for archival history and recovery |
| Versioned working messages and summaries | No | Latest version supplies continuity; older snapshots remain stored |
| Inferred Q&A memories: `synthesized_conversation` | Yes | Expired records stop being eligible for topic recall |
| Archived web episodes: `internet_search_episode` | Yes | Expired observations stop being eligible for topic recall |
| Historical OpenSearch corpus | No | Continues to supply document evidence |

**Neither operation deletes the stored conversation.** Active context can be reduced while persistent storage continues to grow. A statement can remain in working history after a separate topic-memory copy expires. That distinction will matter when you test forgetting.

The two controls are defined in [config.py](src/common/config.py), applied by [the agent](src/news_agent/news_agent.py), and validated by [the REST contract](src/news_agent/rest_contract.py).

## 1. Open Exercise 4 and choose a fresh session

Run every command from the Exercise 4 project directory: the folder containing `client.py`, `ner_server.py`, and `src/`. Use the Python environment and service configuration established earlier in the workshop.

The dataset **`ai_media_dataset_20250911-1000lines.csv` has already been ingested in the previous exercise**. Continue using that corpus and the existing OpenSearch and mem0 services.

Use four terminals:

| Terminal | Purpose |
| --- | --- |
| A | NER server |
| B | Tavily MCP server |
| C | News Agent and lifecycle output |
| D | Questions and inspection commands |

If the Exercise 3 application servers are still running on these ports, stop them with **Ctrl+C** before starting their Exercise 4 counterparts. Run one News Agent process for this walkthrough.

The multiline commands below use Bash/Zsh heredocs, also available in WSL. In PowerShell, save the Python between the `PY` markers into a temporary `.py` file in this project directory and run it with `python filename.py`.

## 2. Start the NER server

In **Terminal A**, run:

```bash
python ner_server.py
```

The default endpoint is `http://127.0.0.1:8000/ner`. Leave the process running.

NER labels new memory content and recall questions with entities such as organizations and people. The memory selector can prefer candidates sharing entities with the current question. This supports topic relevance; it does not determine whether a source is trustworthy. The agent invokes NER throughout this exercise.

Source: [ner_server.py](ner_server.py) and [topic-memory selection](src/common/conversation.py).

## 3. Start the Tavily MCP server

In **Terminal B**, run:

```bash
python -m src.tavily_mcp_server
```

The default MCP endpoint is `http://127.0.0.1:8765/mcp`. Leave it running.

This server exposes `search_technology_news`. The News Agent can call it when a request needs current external evidence. Retrieved snippets retain their source information when archived as memory episodes, and those archived copies receive the expiration policy you will inspect later.

Source: [Tavily MCP server](src/tavily_mcp_server.py).

## 4. Start the Agentic News Agent

In **Terminal C**, run:

```bash
python -m src.news_agent
```

The default chat endpoint is `http://127.0.0.1:9001/v1/chat/completions`. The agent uses the workshop's configured OpenAI services for routing, generation, verification, and compaction.

Keep this terminal visible. Each `memory_operation` line identifies an operation, an action, a status, and details. At the end of each chat request, `memory_summary` reports completed, skipped, and failed events for all five operations.

| Operation | What to look for in this implementation |
| --- | --- |
| Capture | Archived web episodes, Q&A inference submissions, raw conversation, and working snapshots |
| Consolidate | `message_compaction` started, completed, skipped, or failed |
| Retrieve | `restore_working_history`, topic searches, and prompt-memory selection |
| Update | Raw-conversation and working-history head updates; inferred updates if mem0 reports them |
| Forget | `expiration_exclusion` or `exclude_expired_evidence`, with stored records retained |

The summary counts **log events**, not unique memories. Several events can describe different stages of saving one snapshot. A zero count is valid when an operation did not occur.

Sources: [server entry point](src/news_agent/__main__.py) and [lifecycle logging](src/common/logging.py).

## 5. Ask the two AI questions

Run the following commands in **Terminal D**, one at a time.

### Question 1: current work

```bash
python -m src.query --question "What is Anthropic currently working on in Artificial Intelligence (AI)?"
```

Read the answer and its Sources section. In Terminal 3, identify the memory capture and retrieval events. The first approved exchange creates two working messages: the question and answer. With `compact_k=5`, compaction is skipped because the threshold has not been reached.

An approved, audited exchange is submitted to mem0 for Q&A inference. A submission may create a fact, update one, or produce no new fact; it is not a guarantee of one new semantic memory per question. Retrieved web evidence can also be archived as separate episodes.

### Question 2: a specific time period

```bash
python -m src.query --question "What announcements has Anthropic made in Artificial Intelligence (AI) in October 2026?"
```

This starts another client process but uses the same session. The server restores its saved working history before routing the new question. After two successful exchanges, the working history contains four messages and remains at version `0`.

Check the dates of the cited sources against **October 2026**. The corpus is fixed, while MCP supplies current search results. The Tavily adapter uses a relative search window; the requested month remains in the question, but it is not translated into a strict calendar-month filter. Assess the returned evidence rather than expecting an exhaustive list or a fixed answer.

Use the citation labels to distinguish the evidence paths:

| Label | Meaning in the current answer |
| --- | --- |
| `[H1]`, `[H2]`, .. | Historical OpenSearch evidence |
| `[W1]`, `[W2]`, .. | Fresh Tavily MCP evidence |
| `[M1]`, `[M2]`, .. | Eligible recalled memory |

An archived web result recalled later receives an `M` label and retains its original source/date provenance. A memory's creation or retrieval time is not the article's publication date. The selected route and citations can vary with the evidence available.

Before continuing, look for `outcome=completed` and the JSON details `"memory_status":"saved"` and `"history_status":"saved"` in the request's final console summary. A withheld answer or failed save changes the expected progression; count persisted, completed turns rather than counting commands you ran.

Sources: [one-shot client](src/query.py), [capture behavior](src/news_agent/memory.py), [citation assignment](src/news_agent/governance.py), and [Tavily search-window selection](src/common/tavily_client.py).

## 6. Trigger and inspect automatic compaction

Send the third question with the following Python command. It uses the project's HTTP client helper so that you can see the response fields the one-shot CLI normally leaves out.

```bash
python - <<'PY'
import json
from src.common.config import load_settings
from src.query import chat_completion, create_client

settings = load_settings()
with create_client(settings) as client:
    result = chat_completion(
        client,
        model=settings.public_model_name,
        messages=[{
            "role": "user",
            "content": (
                "Which of those Anthropic AI developments should we investigate "
                "next? Use dated sources and identify unanswered questions."
            ),
        }],
        session_id=settings.mem0_session_id,
        compact_k=settings.compact_k,
        memory_expiration=settings.memory_expiration,
    )

print(result["choices"][0]["message"]["content"])
metadata = result["metadata"]
print(json.dumps({
    "status": metadata["status"],
    "memory_status": metadata["memory_status"],
    "history_status": metadata["history_status"],
    "history_snapshot_id": metadata["history_snapshot_id"],
    "version": result["version"],
    "compacted": result["compacted"],
    "returned_messages": result.get("messages", []),
}, indent=2, ensure_ascii=False))
PY
```

Although the request sends one new question, the server restores the earlier working messages. It appends the approved answer **before counting**. The third completed exchange therefore reaches six messages and satisfies `6 >= 5`.

For a fresh session with successful saves and compactions, the progression is:

| Completed exchange | Messages before compaction | Returned user/assistant messages | Version | `compacted` |
| --- | ---: | ---: | ---: | --- |
| 1 | 2 | 2 | 0 | `false` |
| 2 | 4 | 4 | 0 | `false` |
| 3 | 6 | 1 summary | 1 | `true` |
| 4 | 3 | 3 | 1 | `false` |
| 5 | 5 | 1 summary | 2 | `true` |

**Observe:** the full third answer remains in `choices[0].message.content`. The returned `messages` contains one assistant continuity summary. Terminal 3 reports `operation=Consolidate action=message_compaction`, followed by the version transition and input/output message counts.

Read the summary. Does it retain Anthropic as the subject, the October 2026 timeframe, and the unresolved questions? Does it distinguish what the user requested from claims made in earlier answers? Summary wording is generated, so judge its contents as well as its size.

The bundled interactive client replaces its local history with the returned `messages`. Appending the displayed answer again would duplicate the turn. Caller-supplied system/developer messages, if present, are preserved separately in the response and excluded from compaction, its count, and stored working history. The server's own governance instructions remain server-owned.

Sources: [compaction implementation](src/news_agent/news_agent.py), [response fields](src/news_agent/rest_contract.py), and [interactive history replacement](client.py).

## 7. Inspect retained history, then resume after a restart

### Inspect the stored snapshots

When the request has finished, run this read-only inspection in **Terminal D**. It connects to mem0 using the same configured memory owner and session as the agent.

```bash
python - <<'PY'
import json
from src.common.config import load_settings
from src.news_agent.memory import NewsMemory, scoped_memory_settings

settings = load_settings()
memory = NewsMemory(
    scoped_memory_settings(settings, settings.mem0_session_id, None)
)
try:
    history = memory.history()
    latest = history.load_latest()
    if latest is None:
        raise SystemExit("No committed working history exists for this session.")
    print(json.dumps({
        "session_id": settings.mem0_session_id,
        "latest_version": latest.version,
        "latest_messages": latest.messages,
        "raw_message_count": len(memory.load_messages()),
        "snapshots_newest_first": history.list_snapshots(),
    }, indent=2, ensure_ascii=False))
finally:
    memory.close()
PY
```

Immediately after the third successful exchange, expect a version `1` snapshot containing one summary and retained version `0` snapshots, including the six-message snapshot saved before compaction. The raw conversation still contains the three original question/answer pairs.

Each snapshot includes a sequence, parent reference, source audit ID, and content hash. `version` increases when compaction creates a new representation; `sequence` increases for each committed snapshot. These let you trace the summary to prior history. They do not make a generated summary factually authoritative.

Once versioned history exists, the agent does not reinsert the entire raw archive as memory evidence. Doing so would undo the active-context reduction. The archive remains available for storage and inspection.

### Restart and ask a follow-up

Stop the News Agent in **Terminal C** with **Ctrl+C**, then restart it from the same directory and configuration:

```bash
python -m src.news_agent
```

In **Terminal D**, ask:

```bash
python -m src.query --question "Which of those Anthropic research topics should we examine next, and what current sources support that choice?"
```

Find `action=restore_working_history` in Terminal 3. Its JSON details should include `"source":"latest_version"` and version `1`. After this approved exchange, the active working history contains the previous summary, the new question, and the new answer: three messages at version `1`.

Run the read-only inspection again to confirm. The newest snapshot can have `compacted=false` because this particular turn did not compact; its version still records that it extends the earlier compacted history.

**Checkpoint:** a new client process and restarted agent recovered the compacted conversation from persistent memory. Neither process needed to retain the original transcript in RAM.

Sources: [exact snapshot recovery](src/common/message_history.py) and [restore-before-route behavior](src/news_agent/news_agent.py).

## 8. Observe memory expiration without deleting history

Use a separate session for this experiment, while keeping the same memory owner. This separates its working conversation from the main walkthrough. Eligible topic memories can still cross sessions for that user.

### Capture memories with a one-minute lifetime

In **Terminal D**, run:

```bash
python - <<'PY'
import json
from src.common.config import load_settings
from src.query import chat_completion, create_client

settings = load_settings()
session_id = settings.mem0_session_id + "-expiry"
with create_client(settings) as client:
    result = chat_completion(
        client,
        model=settings.public_model_name,
        messages=[{
            "role": "user",
            "content": (
                "What are Anthropic's latest AI research announcements? "
                "Identify the publication dates of supporting sources."
            ),
        }],
        session_id=session_id,
        compact_k=5,
        memory_expiration=1.0,
    )
print(result["choices"][0]["message"]["content"])
print(json.dumps({"expiry_session": session_id, **result["metadata"]}, indent=2))
PY
```

Here `memory_expiration=1.0` means **one minute from each record's write time**, not from the time the response appears. This request override affects its new Q&A/web captures; it does not change existing deadlines or the server's defaults. Confirm that the request completed and both persistence statuses are `saved`.

### Inspect actual deadlines and recall eligibility

Run the following read-only command now, then run it again after the listed expiration deadlines have passed. Avoid sending more questions to this expiry session between the two inspections, so new captures do not obscure the comparison.

```bash
python - <<'PY'
import json
from datetime import datetime, timezone
from src.common.config import load_settings
from src.common.conversation import memory_is_expired
from src.news_agent.memory import NewsMemory, scoped_memory_settings

settings = load_settings()
session_id = settings.mem0_session_id + "-expiry"
scoped = scoped_memory_settings(settings, session_id, None)
memory = NewsMemory(scoped)
try:
    history = memory.history()
    manager = history.manager
    records = []
    for run_id in (
        scoped.mem0_synthesized_run_id,
        scoped.mem0_internet_run_id,
    ):
        records.extend(manager.memory.get_all(run_id=run_id))
    if not records:
        raise SystemExit(
            "No expiring records were found. Inspect the capture request's "
            "memory status and web/Q&A capture events."
        )

    candidates = manager.retrieve_topic_memories("Anthropic AI announcements")
    observed_at = datetime.now(timezone.utc)
    selected_ids = {
        record["id"] for record in candidates
        if not memory_is_expired(record, now=observed_at)
    }
    latest = history.load_latest()
    print(json.dumps({
        "session_id": session_id,
        "observed_at": observed_at.isoformat(),
        "retained_topic_records": [{
            "id": record["id"],
            "kind": record["metadata"].get("memory_kind"),
            "expiration": record["metadata"].get("expiration"),
            "excluded_by_expiration": memory_is_expired(
                record, now=observed_at
            ),
            "selected_and_still_valid": record["id"] in selected_ids,
        } for record in records],
        "raw_message_count": len(memory.load_messages()),
        "working_version": latest.version if latest else None,
    }, indent=2))
finally:
    memory.close()
PY
```

Match the **same IDs** across the two observations. After their deadlines pass, expect:

- The records remain listed under `retained_topic_records`.
- `excluded_by_expiration` is `true`.
- `selected_and_still_valid` is `false`.
- Raw conversation and the working-history version remain available.

The policy excludes a record when `now > expiration`; at exact equality it remains eligible. A slow request can consume a short lifetime before the first inspection. If that happens, the retained expired record still demonstrates logical exclusion. To observe both states, repeat with a longer lifetime such as `3.0` and a fresh suffix, changing `"-expiry"` to `"-expiry-02"` in **both** snippets.

The inspection invokes the real topic-recall path. When an expired candidate is encountered, look for `operation=Forget action=expiration_exclusion` and the JSON detail `"records_retained":true` in **Terminal D**, where the inspection runs. Direct inspection calls use `request_id=-`; HTTP requests produce the per-request summary in Terminal 3. Candidate searches are bounded, so a record may not appear in the search candidates at all. Its stored ID and explicit expiration result are the stronger test.

Absence from top-k alone does not prove expiration: relevance and NER entity selection also influence recall. An answer mentioning an old topic does not prove failure either: the topic may remain in working context or have fresh external evidence. Compare record eligibility separately from generated wording.

Older Exercise 3 topic records without valid expiration metadata are also excluded by this policy. They remain stored, and their raw conversation can still be recovered. The implementation also rechecks deadlines around slow retrieval, generation, and verification stages so a memory can lose evidence eligibility during a request.

Sources: [`memory_is_expired` and topic recall](src/common/conversation.py), [evidence conversion](src/news_agent/memory.py), and [later expiration checks](src/news_agent/news_agent.py).

## Interpret the result before moving on

| Observation | What it establishes |
| --- | --- |
| `compacted=true`, a higher version, and one returned summary | A new compacted working representation was published |
| Older snapshots and raw messages remain inspectable | Compaction preserved the history behind the summary |
| Restart restores `latest_version` | Working continuity comes from persistent state |
| A retained record has expired and is excluded | Forgetting is implemented as retrieval exclusion |
| `Update` events appear | Navigation state changed, or mem0 reported an inferred update; inspect the action to tell which |
| A submitted Q&A produces no new inferred fact | Capture submission and semantic extraction are separate operations |

If the expected counts differ, first use the observed session, version, and persistence statuses to explain them. Reusing a session, an additional completed turn, or a withheld response changes the count. If compaction fails, inspect `history_status` and the retained snapshots: the pre-compaction snapshot is saved before summarization. A successful answer alone does not establish that every later memory write succeeded.

The default audit file is `logs/news-agent-audit.jsonl`, configurable through `AUDIT_LOG_PATH`. It includes `news_agent_memory_write` and `news_agent_message_history` events linked by audit IDs. Console output supplies the immediate operational view; snapshot metadata and the audit trail supply provenance.

This version implements working-history updates and logical expiration. General semantic contradiction reconciliation, importance-based decay, protected-memory retention, and automatic physical deletion of expired records would require additional policy work. The five operation names describe the implemented actions above; they are not a promise that every possible lifecycle capability exists.

## Bonus: Explore lifecycle policies further

### A. Continue with the interactive client

In Terminal 4, run:

```bash
python client.py
```

Enter this question:

```text
For this Anthropic AI research, use detailed technical explanations. Which current developments have dated supporting sources?
```

Then try:

```text
For the next answer, use three concise bullets. Which of those developments should I investigate first?
```

Type `exit` or `quit` to end the client. When a turn compacts, the client prints `[History compacted: version N]`. Observe that the current explicit instruction can refine the response style while older statements remain in history. Inspect the next summary to see how it represents the change.

This demonstrates working-context updates. Mem0 may also update an inferred memory, but a change in answer style does not prove deterministic semantic supersession.

### B. Compare compaction thresholds

Use fresh session names and repeat the three-question sequence with `COMPACT_K=2`, then `COMPACT_K=10`. Set the value in the configuration read by the clients; the bundled clients send it explicitly with each request. If relying on server defaults for other callers, restart the agent after changing those defaults.

With `2`, each completed exchange reaches the threshold. With `10`, the first three exchanges remain below it. Compare the retained detail, number of summary calls, and responsiveness. A smaller threshold summarizes more often; a larger threshold carries more history before summarizing. Message and character counts are useful observations, but neither is a token-cost measurement.

### C. Separate working continuity from cross-session recall

Choose a new `MEMORY_SESSION_ID`, keep the same server-side `MEMORY_USER_ID`, and ask a self-contained Anthropic question. The new session starts a separate working history. Eligible topic memories from earlier sessions may still be recalled, depending on expiration and relevance.

Contrast this with the restart in Step 7, which resumed the same working history. A session boundary and a memory-owner boundary serve different purposes.

### D. Follow provenance through a compaction

Repeat the Step 7 inspection. Pick a compacted snapshot, locate its parent and `audit_id`, and inspect the related entries in the audit file. Compare the summary with the retained raw conversation. Identify a detail omitted from active context and an explicit user statement preserved in it.

Ask: what should a production policy keep in exact form even when the conversation is summarized? Consider accepted decisions, unresolved commitments, and user corrections. The archive enables that analysis, while the current compactor produces a bounded continuity summary.

## Finish: What did this exercise demonstrate?

You extended the same News Agent with two explicit lifecycle policies and observed their effects through stored state:

1. **Capture:** approved, audited exchanges produce raw conversation, Q&A inference submissions, and working snapshots; retrieved web evidence can be archived as episodes.
2. **Consolidate:** a message-count threshold creates a compacted working version while preserving earlier snapshots.
3. **Retrieve:** the agent restores the current working version and selects eligible topic memories alongside historical and current external evidence.
4. **Update:** working-history navigation advances as turns and summaries are published; mem0 may independently report inferred-memory updates.
5. **Forget:** expired topic memories leave the normal recall path while their records and the conversation archive remain available.

Before leaving the exercise, confirm that you observed a compaction, recovered its version after a restart, and inspected an expired record that remained stored. These observations distinguish persistent storage, active context, and retrieval eligibility... the three concerns a long-running memory architecture must manage separately.

When finished, stop the interactive client, if open, and the three exercise servers with **Ctrl+C**. Your retained conversation and version history remain in mem0.
