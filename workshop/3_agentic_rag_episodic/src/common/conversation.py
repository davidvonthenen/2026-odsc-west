"""Higher-level conversation persistence and episodic-memory management."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..ner_client import NERServiceClient, NERServiceError

from .config import Settings
from .mem0_client import Mem0Client


# Evidence authority is claim-specific, not a ranking of storage backends.
# This policy changes answer construction only; it never rewrites stored memories.
SOURCE_AUTHORITY_POLICY = (
    "Resolve source authority for each claim, not by storage backend or retrieval score. "
    "Apply these rules:\n"
    "1. User and task state: the current user's explicit preferences, constraints, and "
    "corrections take priority. Next use applicable explicit user statements in the "
    "prior conversation or recorded source user messages attached to memories; prefer "
    "the latest clear statement about the same project, task, and time period. A "
    "temporary override does not revoke a durable preference. External documents and "
    "web pages do not override the user's own preferences or decisions.\n"
    "2. Remembered state: use synthesized memories as supporting context, not as "
    "verified user statements. They may contain assistant-generated claims. Attribute "
    "a preference or decision to the user only when explicit user text supports it. "
    "A question, quotation, or hypothetical in that text is not user confirmation. "
    "When provenance is missing or ambiguous, qualify the recollection or ask for "
    "confirmation instead of inventing a settled decision.\n"
    "3. Conversation history: use role-labeled turns as the record of what was said "
    "and for continuity. Prior assistant answers are not independent proof of external "
    "facts, user decisions, or completed actions. An exact user statement takes "
    "precedence over a conflicting model-generated summary of that statement.\n"
    "4. External facts: prefer direct, relevant source evidence over unsupported "
    "conversation claims or model-generated summaries. For a question about a specific "
    "document, that document governs what it says. For changing world facts, compare "
    "source authority and the publication/effective date relevant to the question. "
    "Neither OpenSearch nor live web search is automatically authoritative. Archived "
    "internet memories remain external evidence, not user testimony; an archived "
    "primary source can outweigh an undated or weaker fresh search result.\n"
    "5. Time and conflicts: retrieval time and memory created_at/updated_at are not "
    "publication dates or proof that a claim is current. Do not infer chronology from "
    "session IDs, similarity scores, or prompt order. Preserve the requested historical "
    "time frame. For unresolved conflicts, cite the competing evidence and explain "
    "the uncertainty rather than silently choosing a winner."
)


class ConversationError(RuntimeError):
    """Raised when a stored conversation cannot be reconstructed safely."""


class ConversationManager:
    """Store raw conversations, inferred memories, and internet episodes in Mem0."""

    def __init__(
        self,
        settings: Settings,
        memory: Mem0Client,
        ner: NERServiceClient | None = None,
    ) -> None:
        self.settings = settings
        self.loaded_records: List[Dict[str, Any]] = []
        self.write_metadata: Dict[str, Any] = {}
        self.memory = memory
        self.ner = ner or NERServiceClient(
            url=settings.ner_url,
            timeout=settings.ner_timeout,
        )

    def _entities(self, text: str) -> List[str]:
        """Return stable ``LABEL:name`` values from the unchanged NER client."""
        response = self.ner.post_ner(text)
        pairs = response.get("entity_pairs")
        if not isinstance(pairs, list):
            raise NERServiceError("NER must return an entity_pairs list, including when it is empty.")

        seen: set[str] = set()
        entities: List[str] = []
        for pair in pairs:
            if not isinstance(pair, dict):
                raise NERServiceError("NER returned an invalid entity pair.")
            if not isinstance(pair.get("name"), str) or not isinstance(pair.get("label"), str):
                raise NERServiceError("NER entity names and labels must be strings.")
            name = pair["name"].strip().lower()
            label = pair["label"].strip().upper()
            if not name or not label:
                raise NERServiceError("NER entity pairs must have a name and label.")
            entity = f"{label}:{name}"
            if entity in seen:
                continue
            seen.add(entity)
            entities.append(entity)
        return entities

    def _linked_metadata(
        self,
        *,
        event_ids: Sequence[str],
        index: int,
        entities: Sequence[str],
        memory_kind: str,
        extra: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Build one-based linkage metadata for a chunk or episode."""
        metadata: Dict[str, Any] = {
            "user_id": self.settings.mem0_user_id,
            "session_id": self.settings.mem0_session_id,
            "chunk_index": index + 1,
            "chunk_count": len(event_ids),
            "previous_chunk_id": event_ids[index - 1] if index > 0 else None,
            "next_chunk_id": event_ids[index + 1]
            if index + 1 < len(event_ids)
            else None,
            "entities": list(entities),
            "ner_processed": True,
            "event_id": event_ids[index],
            "memory_kind": memory_kind,
        }
        metadata.update(self.write_metadata)
        if extra:
            metadata.update(extra)
        return metadata

    def _conversation_chunks(
        self, messages: Sequence[Mapping[str, str]]
    ) -> List[Dict[str, Any]]:
        """Split message content without losing role, order, or reconstruction data."""
        max_chars = max(1, int(self.settings.conversation_chunk_chars))
        chunks: List[Dict[str, Any]] = []
        for message_index, message in enumerate(messages):
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "")
            fragment_count = max(1, math.ceil(len(content) / max_chars))
            for fragment_index in range(fragment_count):
                start = fragment_index * max_chars
                chunks.append(
                    {
                        "role": role,
                        "content": content[start : start + max_chars],
                        "message_index": message_index,
                        "fragment_index": fragment_index,
                        "fragment_count": fragment_count,
                    }
                )
        return chunks

    @staticmethod
    def _snapshot_id(messages: Sequence[Mapping[str, str]]) -> str:
        """Create a stable identifier for one complete conversation snapshot."""
        serialized = json.dumps(
            [
                {
                    "role": str(message.get("role") or ""),
                    "content": str(message.get("content") or ""),
                }
                for message in messages
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _deduplicated_by_event(
        memories: Iterable[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Keep one record per deterministic event ID."""
        events: Dict[str, Mapping[str, Any]] = {}
        anonymous: List[Mapping[str, Any]] = []
        for memory in memories:
            metadata = memory.get("metadata") or {}
            event_id = str(metadata.get("event_id") or "")
            if event_id:
                events[event_id] = memory
            else:
                anonymous.append(memory)
        return list(events.values()) + anonymous

    @property
    def _checkpoint_run_id(self) -> str:
        """Keep the mutable head separate from append-only conversation chunks."""
        return f"{self.settings.mem0_complete_run_id}-head"

    def _checkpoint(self) -> tuple[str, Dict[str, Any]] | None:
        """Read the one remote checkpoint for this user/session, without a cache."""
        records = self.memory.get_all(run_id=self._checkpoint_run_id)
        if not records:
            return None
        if len(records) != 1:
            raise ConversationError(
                "Multiple conversation checkpoints exist. Use one writer per user/session."
            )
        record = records[0]
        metadata = record.get("metadata") or {}
        try:
            value = json.loads(record["memory"])
            valid = (
                isinstance(value, dict)
                and value.get("schema") == 1
                and value.get("user_id") == self.settings.mem0_user_id
                and value.get("session_id") == self.settings.mem0_session_id
                and isinstance(value.get("first_memory_id"), str)
                and bool(value["first_memory_id"])
                and isinstance(value.get("snapshot_id"), str)
                and bool(value["snapshot_id"])
                and type(value.get("chunk_count")) is int
                and value["chunk_count"] > 0
                and type(value.get("message_count")) is int
                and value["message_count"] > 0
                and metadata.get("memory_kind") == "conversation_checkpoint"
            )
            if not valid or not record.get("id"):
                raise ValueError("Invalid checkpoint fields")
        except (TypeError, ValueError, KeyError) as exc:
            raise ConversationError("The remote conversation checkpoint is invalid.") from exc
        return str(record["id"]), value

    def _checkpoint_records(self, checkpoint: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Traverse exact memory IDs; semantic search is never used for recovery."""
        records: List[Dict[str, Any]] = []
        seen: set[str] = set()
        memory_id = checkpoint["first_memory_id"]
        for index in range(checkpoint["chunk_count"]):
            if not isinstance(memory_id, str) or not memory_id or memory_id in seen:
                raise ConversationError("The snapshot chain is incomplete or contains a cycle.")
            seen.add(memory_id)
            record = self.memory.get(memory_id, run_id=self.settings.mem0_complete_run_id)
            metadata = record.get("metadata") or {}
            if (
                metadata.get("memory_kind") != "conversation_chain"
                or metadata.get("session_id") != self.settings.mem0_session_id
                or metadata.get("snapshot_id") != checkpoint["snapshot_id"]
                or metadata.get("chunk_index") != index + 1
                or metadata.get("chunk_count") != checkpoint["chunk_count"]
                or metadata.get("conversation_message_count") != checkpoint["message_count"]
            ):
                raise ConversationError("A checkpoint points to inconsistent snapshot metadata.")
            records.append(record)
            memory_id = metadata.get("next_memory_id")
        if memory_id is not None:
            raise ConversationError("The snapshot has more chunks than its checkpoint declares.")
        return records

    @staticmethod
    def _created_id(response: Any) -> str:
        results = Mem0Client._results(response)
        if len(results) != 1 or not results[0].get("id"):
            raise ConversationError("Mem0 did not return an ID for a raw snapshot record.")
        return str(results[0]["id"])

    def save_complete_conversation(
        self, messages: Sequence[Mapping[str, str]]
    ) -> int:
        """Append a lossless snapshot, then publish its remote checkpoint last.

        Chunks are inserted in reverse so each can reference the next server ID.
        Earlier snapshots are retained. A failed chunk write leaves the previous
        checkpoint untouched; retrying may leave unreferenced duplicate chunks.
        The REST API has no compare-and-swap, so a session has one writer.
        """
        chunks = self._conversation_chunks(messages)
        if not chunks:
            return 0

        checkpoint = self._checkpoint()
        snapshot_id = self._snapshot_id(messages)
        if checkpoint is not None and checkpoint[1]["snapshot_id"] == snapshot_id:
            # Do not acknowledge a duplicate if its stored chain was damaged.
            self.load_complete_conversation()
            return 0

        run_id = self.settings.mem0_complete_run_id
        event_ids = [
            f"evt-{self.settings.mem0_session_id}-complete-{snapshot_id}-{index + 1}"
            for index in range(len(chunks))
        ]
        saved_at = datetime.now(timezone.utc).isoformat()
        next_memory_id = None
        snapshot_entities: set[str] = set()
        for index in range(len(chunks) - 1, -1, -1):
            chunk = chunks[index]
            entities = self._entities(f"{chunk['role']}: {chunk['content']}")
            snapshot_entities.update(entities)
            metadata = self._linked_metadata(
                event_ids=event_ids,
                index=index,
                entities=entities,
                memory_kind="conversation_chain",
                extra={
                    "snapshot_id": snapshot_id,
                    "saved_at": saved_at,
                    "conversation_message_count": len(messages),
                    "message_index": chunk["message_index"],
                    "fragment_index": chunk["fragment_index"],
                    "fragment_count": chunk["fragment_count"],
                    "next_memory_id": next_memory_id,
                },
            )
            next_memory_id = self._created_id(self.memory.add(
                json.dumps(chunk, ensure_ascii=False, separators=(",", ":")),
                run_id=run_id,
                metadata=metadata,
                infer=False,
            ))

        head = {
            "schema": 1,
            "user_id": self.settings.mem0_user_id,
            "session_id": self.settings.mem0_session_id,
            "snapshot_id": snapshot_id,
            "first_memory_id": next_memory_id,
            "chunk_count": len(chunks),
            "message_count": len(messages),
        }
        # Verify the new chain before replacing the only published checkpoint.
        stored = self._checkpoint_records(head)
        for record, chunk in zip(stored, chunks):
            expected = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
            if record.get("memory") != expected:
                raise ConversationError("The new snapshot content failed its read-back check.")
        serialized = json.dumps(head, ensure_ascii=False, separators=(",", ":"))
        if checkpoint is None:
            self.memory.add(
                serialized,
                run_id=self._checkpoint_run_id,
                metadata={
                    **self.write_metadata,
                    "session_id": self.settings.mem0_session_id,
                    "memory_kind": "conversation_checkpoint",
                    "entities": sorted(snapshot_entities),
                    "ner_processed": True,
                },
                infer=False,
            )
        else:
            self.memory.update(
                checkpoint[0], run_id=self._checkpoint_run_id, text=serialized,
                metadata={**self.write_metadata, "entities": sorted(snapshot_entities), "ner_processed": True},
            )
        published = self._checkpoint()
        if published is None or published[1] != head:
            raise ConversationError("The new conversation checkpoint could not be verified.")
        return len(chunks)

    def load_complete_conversation(self) -> List[Dict[str, str]]:
        """Load and reconstruct the newest complete conversation snapshot."""
        self.loaded_records = []
        checkpoint = self._checkpoint()
        if checkpoint is not None:
            memories = self._checkpoint_records(checkpoint[1])
        else:
            # Backward-compatible read for pre-checkpoint snapshots already in
            # this server. A full legacy listing fails rather than truncates.
            memories = self._deduplicated_by_event(
                self.memory.get_all(
                    run_id=self.settings.mem0_complete_run_id,
                    metadata_filters={"memory_kind": "conversation_chain"},
                )
            )
        if not memories:
            return []

        snapshots: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for memory in memories:
            metadata = memory.get("metadata") or {}
            snapshot_id = str(metadata.get("snapshot_id") or "")
            if snapshot_id:
                snapshots[snapshot_id].append(memory)
        if not snapshots:
            return []

        complete_snapshots: List[tuple[str, List[Mapping[str, Any]]]] = []
        for snapshot_id, records in snapshots.items():
            records.sort(
                key=lambda item: int(
                    (item.get("metadata") or {}).get("chunk_index", 0)
                )
            )
            expected_count = int(
                (records[0].get("metadata") or {}).get("chunk_count", 0)
            )
            indexes = [
                int((record.get("metadata") or {}).get("chunk_index", 0))
                for record in records
            ]
            if expected_count == len(records) and indexes == list(
                range(1, expected_count + 1)
            ):
                complete_snapshots.append((snapshot_id, records))

        if not complete_snapshots:
            raise ConversationError("No complete conversation snapshot is available.")

        def snapshot_order(item: tuple[str, List[Mapping[str, Any]]]) -> tuple[int, str]:
            records = item[1]
            metadata = records[0].get("metadata") or {}
            return (
                int(metadata.get("conversation_message_count") or 0),
                str(metadata.get("saved_at") or ""),
            )

        selected_snapshot_id, stored = max(complete_snapshots, key=snapshot_order)

        fragments: Dict[int, List[tuple[int, str, str, int]]] = defaultdict(list)
        for memory in stored:
            raw = memory.get("memory")
            if not isinstance(raw, str):
                raise ConversationError("Conversation chunk has no string memory payload.")
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConversationError("Conversation chunk contains invalid JSON.") from exc

            try:
                metadata = memory.get("metadata") or {}
                for key in ("message_index", "fragment_index", "fragment_count"):
                    if type(chunk[key]) is not int or chunk[key] != metadata.get(key):
                        raise ValueError(f"Invalid {key}")
                if (chunk["message_index"] < 0 or chunk["fragment_count"] < 1
                        or not 0 <= chunk["fragment_index"] < chunk["fragment_count"]
                        or not isinstance(chunk["role"], str)
                        or not isinstance(chunk["content"], str)):
                    raise ValueError("Invalid conversation fragment")
                message_index = chunk["message_index"]
                fragment_index = chunk["fragment_index"]
                fragments[message_index].append(
                    (fragment_index, chunk["role"], chunk["content"], chunk["fragment_count"])
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ConversationError("Conversation fragment metadata is inconsistent.") from exc

        expected_messages = int((stored[0].get("metadata") or {}).get("conversation_message_count", 0))
        if sorted(fragments) != list(range(expected_messages)):
            raise ConversationError("Conversation message indexes are incomplete.")
        messages: List[Dict[str, str]] = []
        for message_index in sorted(fragments):
            ordered = sorted(fragments[message_index], key=lambda item: item[0])
            counts = {item[3] for item in ordered}
            if counts != {len(ordered)} or [item[0] for item in ordered] != list(range(len(ordered))):
                raise ConversationError("Conversation fragments are missing or duplicated.")
            roles = {item[1] for item in ordered}
            if len(roles) != 1:
                raise ConversationError(
                    f"Conversation message {message_index} contains conflicting roles."
                )
            messages.append(
                {
                    "role": ordered[0][1],
                    "content": "".join(item[2] for item in ordered),
                }
            )
        if self._snapshot_id(messages) != selected_snapshot_id:
            raise ConversationError("The reconstructed conversation failed its content-hash check.")
        self.loaded_records = [dict(record) for record in stored]
        return messages

    @staticmethod
    def _turn_event_id(session_id: str, question: str, answer: str) -> str:
        """Create a stable event ID for one inferred user/assistant turn."""
        digest = hashlib.sha256(f"{question}\0{answer}".encode("utf-8")).hexdigest()[:16]
        return f"evt-{session_id}-synthesized-{digest}"

    def save_synthesized_turn(self, question: str, answer: str) -> int:
        """Store one user/assistant turn with ``infer=True``."""
        run_id = self.settings.mem0_synthesized_run_id
        event_id = self._turn_event_id(self.settings.mem0_session_id, question, answer)
        if self.memory.has_event(event_id, run_id=run_id, query=question):
            return 0

        entities = self._entities(f"user: {question}\nassistant: {answer}")
        metadata = self._linked_metadata(
            event_ids=[event_id],
            index=0,
            entities=entities,
            memory_kind="synthesized_conversation",
            extra={"question": question, "provenance": "conversation_turn"},
        )
        self.memory.add(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            run_id=run_id,
            metadata=metadata,
            infer=True,
        )
        return 1

    @staticmethod
    def _internet_event_id(session_id: str, question: str, url: str, index: int, payload: str = "") -> str:
        """Create a stable episode ID for repeated internet searches."""
        identity = f"{question}\0{url or index}\0{payload}".encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()[:16]
        return f"evt-{session_id}-web-{digest}"

    def save_internet_results(
        self,
        question: str,
        results: Sequence[Mapping[str, Any]],
    ) -> int:
        """Run every result through NER and append new raw web episodes to Mem0."""
        if not results:
            return 0

        run_id = self.settings.mem0_internet_run_id
        event_ids = [
            self._internet_event_id(
                self.settings.mem0_session_id,
                question,
                str(result.get("url") or ""),
                index,
                json.dumps({key: result.get(key) for key in (
                    "title", "content", "provider", "published_at"
                )}, sort_keys=True, ensure_ascii=False),
            )
            for index, result in enumerate(results)
        ]
        added = 0
        for index, result in enumerate(results):
            title = str(result.get("title") or "")
            url = str(result.get("url") or "")
            content = str(result.get("content") or "")
            provider = str(result.get("provider") or "unknown")
            entities = self._entities(f"{question}\n{title}\n{content}")

            event_id = event_ids[index]
            if self.memory.has_event(event_id, run_id=run_id, query=question):
                continue

            episode = {
                "query": question,
                "provider": provider,
                "title": title,
                "url": url,
                "content": content,
                "score": result.get("score"),
                "published_at": result.get("published_at"),
                "retrieved_at": result.get("retrieved_at"),
                "source_id": result.get("source_id"),
                "tool": result.get("tool"),
            }
            metadata = self._linked_metadata(
                event_ids=event_ids,
                index=index,
                entities=entities,
                memory_kind="internet_search_episode",
                extra={
                    "provider": provider,
                    "query": question,
                    "source_title": title,
                    "source_url": url,
                    "provenance": "internet_search",
                    "published_at": result.get("published_at"),
                    "retrieved_at": result.get("retrieved_at"),
                    "source_id": result.get("source_id"),
                    "tool": result.get("tool"),
                },
            )
            self.memory.add(
                json.dumps(episode, ensure_ascii=False, separators=(",", ":")),
                run_id=run_id,
                metadata=metadata,
                infer=False,
            )
            added += 1
        return added

    @staticmethod
    def _memory_score(memory: Mapping[str, Any]) -> float:
        """Convert an optional Mem0 score to a sortable value."""
        try:
            return float(memory.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def retrieve_topic_memories(self, question: str) -> List[Dict[str, Any]]:
        """Recall topic memories across sessions and prefer shared NER entities."""
        question_entities = set(self._entities(question))
        candidate_limit = max(self.settings.mem0_top_k * 3, self.settings.mem0_top_k)

        combined: Dict[str, Dict[str, Any]] = {}
        for memory_kind in (
            "synthesized_conversation",
            "internet_search_episode",
        ):
            for candidate in self.memory.search(
                question,
                metadata_filters={"memory_kind": memory_kind},
                top_k=candidate_limit,
            ):
                metadata = candidate.get("metadata") or {}
                key = str(
                    candidate.get("id")
                    or metadata.get("event_id")
                    or f"{memory_kind}:{candidate.get('memory', '')}"
                )
                current = combined.get(key)
                if current is None or self._memory_score(candidate) > self._memory_score(
                    current
                ):
                    combined[key] = candidate

        candidates = sorted(
            combined.values(),
            key=self._memory_score,
            reverse=True,
        )[:candidate_limit]

        matching: List[Dict[str, Any]] = []
        for candidate in candidates:
            metadata = candidate.get("metadata") or {}
            stored_entities = {
                str(entity) for entity in metadata.get("entities", []) if entity
            }
            entity_matches = sorted(question_entities.intersection(stored_entities))
            candidate["entity_matches"] = entity_matches
            if entity_matches:
                matching.append(candidate)

        selected = matching if question_entities and matching else candidates
        return selected[: self.settings.mem0_top_k]


__all__ = ["ConversationError", "ConversationManager", "SOURCE_AUTHORITY_POLICY"]
