"""Bridge the ported Mem0 conversation manager to the governed evidence registry."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Any, Callable

from ..common.config import Settings
from ..common.conversation import ConversationManager
from ..common.mem0_client import Mem0Client, Mem0ServiceError
from ..ner_client import NERServiceClient
from .models import EvidenceItem
from .utils import neutralize_stored_citations, sha256_text, utc_now_iso


@dataclass
class MemoryRecall:
    """Lossless history for storage and a bounded evidence view for the models."""

    messages: list[dict[str, str]]
    evidence: list[EvidenceItem]
    trace: dict[str, Any]


def scoped_memory_settings(settings: Settings, session_id: str, user_id: str | None) -> Settings:
    """Isolate this specialist; caller metadata is not an authenticated identity."""
    user = settings.mem0_user_id if user_id is None else user_id
    if not isinstance(user, str) or not user.strip():
        raise ValueError("A nonempty memory user identity is required")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A nonempty context/session ID is required")
    if (settings.mem0_top_k < 1 or settings.conversation_chunk_chars < 1
            or settings.memory_context_max_chars < 1 or settings.ner_timeout <= 0):
        raise ValueError("Memory limits and NER timeout must be positive")
    return replace(
        settings,
        mem0_user_id=f"news-agent:{user.strip()}",
        mem0_session_id=session_id.strip(),
    )


async def memory_call(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run blocking SDK/NER work off-loop and drain it before cancellation cleanup.

    Cancelling an asyncio.to_thread await does not stop its HTTP worker. Keeping
    that worker alive until it exits prevents close/write and session-lock races.
    """
    task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
            if task.cancelled():
                raise
            if task.done():
                # Consume a simultaneous worker exception before propagating the
                # caller cancellation. No worker is left owning the transport.
                try:
                    task.result()
                except Exception:
                    pass
                raise
            # A second cancellation must not release the lock or close the HTTP
            # session while the first cancelled operation is still writing.
        except Exception:
            if cancelled:
                raise asyncio.CancelledError() from None
            raise
    if cancelled:
        raise asyncio.CancelledError()
    return result



class NewsMemory:
    """One lazily connected Mem0/NER session per authorized agent invocation."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Mem0Client | None = None,
        ner: NERServiceClient | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.ner = ner
        self.manager: ConversationManager | None = None

    def _connect(self) -> ConversationManager:
        if self.manager is None:
            if self.client is None:
                self.client = Mem0Client(self.settings)
            if self.ner is None:
                self.ner = NERServiceClient(
                    url=self.settings.ner_url, timeout=self.settings.ner_timeout,
                )
            self.manager = ConversationManager(self.settings, self.client, self.ner)
        return self.manager

    def _evidence(self, record: dict[str, Any], retrieved_at: str) -> EvidenceItem:
        metadata = dict(record.get("metadata") or {})
        kind = metadata.get("memory_kind")
        if metadata.get("ner_processed") is not True or not isinstance(metadata.get("entities"), list):
            raise Mem0ServiceError("Recalled memory lacks required NER labeling")
        raw = record.get("memory")
        if not isinstance(raw, str) or not raw.strip() or not record.get("id"):
            raise Mem0ServiceError("Recalled memory has no usable text or ID")
        title = "Model-extracted conversation memory"
        url = None
        published_at = None
        prompt_metadata = {
            key: metadata[key] for key in (
                "memory_kind", "session_id", "event_id", "snapshot_id", "saved_at",
                "message_index", "fragment_index", "fragment_count", "chunk_index",
                "chunk_count", "entities", "ner_processed", "provenance", "question",
                "source_audit_id", "source_verification_status", "provider", "tool",
            ) if key in metadata
        }
        prompt_metadata.update({
            "memory_id": str(record["id"]),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "entity_matches": record.get("entity_matches", []),
        })
        if kind == "internet_search_episode":
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
                raise Mem0ServiceError("Archived internet episode has an invalid payload")
            title = str(payload.get("title") or metadata.get("source_title") or "Archived news source")
            url = str(payload.get("url") or metadata.get("source_url") or "") or None
            published_at = payload.get("published_at") or metadata.get("published_at")
            prompt_metadata["original_retrieved_at"] = payload.get("retrieved_at") or metadata.get("retrieved_at")
            prompt_metadata["authority"] = "archived_external_evidence_not_user_testimony"
            # Preserve query/provider/source provenance and allocate the snippet
            # budget to the actual article content rather than its JSON header.
            prompt_metadata["original_query"] = payload.get("query")
            raw = payload["content"]
        elif kind == "conversation_chain":
            payload = json.loads(raw)
            role = payload.get("role")
            if role not in {"user", "assistant"} or not isinstance(payload.get("content"), str):
                raise Mem0ServiceError("Conversation evidence has an invalid role or payload")
            title = f"Recorded {role} turn, message {payload['message_index'] + 1}"
            prompt_metadata["role"] = role
            prompt_metadata["authority"] = "record_of_what_was_said_not_independent_external_proof"
            raw = payload["content"]
        elif kind == "synthesized_conversation":
            prompt_metadata["authority"] = "model_extracted_context_not_verified_user_testimony"
        else:
            raise Mem0ServiceError("Unsupported memory kind in the evidence registry")

        return EvidenceItem(
            source_kind="memory", source_id=f"mem0:{record['id']}",
            title=neutralize_stored_citations(title), text=neutralize_stored_citations(raw),
            url=url, published_at=str(published_at) if published_at else None,
            retrieved_at=retrieved_at, score=record.get("score"),
            chunk_index=metadata.get("chunk_index"), metadata=prompt_metadata,
        )

    def recall(self, question: str) -> MemoryRecall:
        manager = self._connect()
        # This calls NER even when the remote memory store contains no records.
        memories = manager.retrieve_topic_memories(question)
        messages = manager.load_complete_conversation()
        now = utc_now_iso()
        topic = [self._evidence(record, now) for record in memories]
        history = [self._evidence(record, now) for record in manager.loaded_records]
        budget = self.settings.memory_context_max_chars
        topic_evidence: list[EvidenceItem] = []
        # Reserve half of the text budget for recent exact conversation when it
        # exists. The full conversation remains intact in MemoryRecall.messages.
        topic_budget = budget // 2 if history else budget
        for item in topic:
            limit = min(self.settings.evidence_snippet_max_chars, topic_budget)
            if limit <= 0:
                break
            text = item.text[:limit]
            metadata = {**item.metadata, "content_truncated": len(text) < len(item.text)}
            topic_evidence.append(item.model_copy(update={"text": text, "metadata": metadata}))
            topic_budget -= len(text)
            budget -= len(text)
        recent: list[EvidenceItem] = []
        for item in reversed(history):
            if budget <= 0:
                break
            # Keep whole stored fragments where possible, newest fragments first.
            text = item.text[-budget:]
            metadata = {**item.metadata, "content_truncated": len(text) < len(item.text)}
            recent.append(item.model_copy(update={"text": text, "metadata": metadata}))
            budget -= len(text)
        evidence = [*topic_evidence, *reversed(recent)]
        return MemoryRecall(messages=messages, evidence=evidence, trace={
            "status": "recalled",
            "user_id_sha256": sha256_text(self.settings.mem0_user_id),
            "session_id": self.settings.mem0_session_id,
            "backend_provider": self.client.backend_provider,
            "topic_count": len(topic_evidence),
            "topic_candidates": len(topic),
            "conversation_message_count": len(messages),
            "conversation_chunks_in_prompt": len(recent),
            "conversation_chunks_stored": len(history),
            "prompt_text_chars": sum(len(item.text) for item in evidence),
            "context_truncated": len(recent) < len(history) or len(topic_evidence) < len(topic) or any(
                item.metadata.get("content_truncated") for item in evidence
            ),
            "ner_required": True,
        })

    def persist(
        self,
        question: str,
        answer: str | None,
        recall: MemoryRecall,
        evidence: list[EvidenceItem],
        audit_id: str,
        verification_status: str,
    ) -> dict[str, Any]:
        """Archive source results and save only an audited, releasable answer.

        There is no transaction across Mem0 writes. A failure may leave archived
        sources or a complete checkpoint; it must not be reported as full success.
        """
        manager = self._connect()
        manager.write_metadata = {
            "source_audit_id": audit_id,
            "source_verification_status": verification_status,
        }
        web_results = [
            {
                "title": item.title, "url": item.url, "content": item.text,
                "score": item.score, "provider": item.metadata.get("provider", "tavily"),
                "published_at": item.published_at, "retrieved_at": item.retrieved_at,
                "source_id": item.source_id, "tool": item.metadata.get("tool"),
            }
            for item in evidence if item.source_kind == "realtime"
        ]
        counts = {"internet_episodes": manager.save_internet_results(question, web_results),
                  "conversation_chunks": 0, "synthesized_turns_submitted": 0}
        if answer is not None:
            messages = [*recall.messages, {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}]
            counts["conversation_chunks"] = manager.save_complete_conversation(messages)
            counts["synthesized_turns_submitted"] = manager.save_synthesized_turn(question, answer)
        return {"status": "saved", **counts}

    def close(self) -> None:
        try:
            if self.client is not None:
                self.client.close()
        finally:
            if self.ner is not None:
                self.ner.close()
