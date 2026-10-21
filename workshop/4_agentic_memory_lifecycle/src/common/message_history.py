"""Versioned working messages with immutable snapshots and a remote head pointer.

Only the navigation pointer is updated. Chunks and commit manifests are append-only,
including the uncompacted snapshot preceding every compaction. Exact memory IDs
make recovery independent of vector ranking and the OSS list limit. As with the
existing conversation checkpoint, callers must serialize writers for a session;
the Mem0 REST API does not provide compare-and-swap across application workers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .conversation import ConversationError, ConversationManager
from .logging import memory_event


@dataclass(frozen=True)
class HistorySnapshot:
    messages: list[dict[str, str]]
    version: int
    sequence: int
    snapshot_id: str
    audit_id: str
    compacted: bool
    saved_at: str
    content_sha256: str
    memory_id: str
    parent_memory_id: str | None


class MessageHistory:
    """Persist the active non-system messages separately from raw conversation."""

    def __init__(self, manager: ConversationManager) -> None:
        self.manager = manager
        self.settings = manager.settings
        self.memory = manager.memory
        self.run_id = f"{self.settings.mem0_session_id}-messages"
        self.chunk_run_id = f"{self.run_id}-chunks"
        self.head_run_id = f"{self.run_id}-head"

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if (role not in {"user", "assistant"} or not isinstance(content, str)
                    or ("name" in message and not isinstance(message["name"], str))):
                raise ConversationError("Working history requires user/assistant text messages.")
            normalized.append({"role": role, "content": content,
                               **({"name": message["name"]} if "name" in message else {})})
        if not normalized:
            raise ConversationError("A history snapshot must contain at least one message.")
        return normalized

    def _metadata(self, kind: str, text: str, **extra: Any) -> dict[str, Any]:
        return {
            **self.manager.write_metadata,
            "session_id": self.settings.mem0_session_id,
            "memory_kind": kind,
            "entities": self.manager._entities(text),
            "ner_processed": True,
            **extra,
        }

    def _payload(self, record: Mapping[str, Any], kind: str) -> dict[str, Any]:
        metadata = record.get("metadata") or {}
        try:
            value = json.loads(record["memory"])
            if (not record.get("id") or not isinstance(value, dict)
                    or value.get("schema") != 1
                    or value.get("user_id") != self.settings.mem0_user_id
                    or value.get("session_id") != self.settings.mem0_session_id
                    or metadata.get("session_id") != self.settings.mem0_session_id
                    or metadata.get("memory_kind") != kind
                    or metadata.get("ner_processed") is not True
                    or not isinstance(metadata.get("entities"), list)):
                raise ValueError("Invalid history record fields")
            return value
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversationError("A message-history record is invalid.") from exc

    def _head(self) -> tuple[str, dict[str, Any]] | None:
        records = self.memory.get_all(run_id=self.head_run_id)
        if not records:
            memory_event("Retrieve", "message_history_head", head_found=False)
            return None
        if len(records) != 1:
            raise ConversationError("Multiple message-history heads exist; use one session writer.")
        value = self._payload(records[0], "message_history_head")
        if not isinstance(value.get("memory_id"), str) or not value["memory_id"]:
            raise ConversationError("The message-history head has no commit ID.")
        memory_event("Retrieve", "message_history_head", head_found=True)
        return str(records[0]["id"]), value

    def _commit(self, memory_id: str) -> dict[str, Any]:
        record = self.memory.get(memory_id, run_id=self.run_id)
        value = self._payload(record, "message_history_commit")
        metadata = record.get("metadata") or {}
        integer_fields = {"version": 0, "sequence": 1, "chunk_count": 1, "message_count": 1}
        if any(type(value.get(key)) is not int or value[key] < minimum
               for key, minimum in integer_fields.items()):
            raise ConversationError("A history commit has invalid counters.")
        if (type(value.get("compacted")) is not bool
                or any(not isinstance(value.get(key), str) or not value[key]
                       for key in ("snapshot_id", "first_memory_id", "audit_id", "saved_at"))
                or not isinstance(value.get("content_sha256"), str)
                or len(value["content_sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in value["content_sha256"])
                or any(metadata.get(key) != value[key]
                       for key in ("version", "sequence", "snapshot_id", "content_sha256", "compacted", "saved_at"))
                or metadata.get("source_audit_id") != value["audit_id"]
                or (value["compacted"] and value["message_count"] != 1)):
            raise ConversationError("A history commit has inconsistent metadata.")
        parent = value.get("parent_memory_id")
        if value["sequence"] == 1:
            if parent is not None or value["version"] != 0 or value["compacted"]:
                raise ConversationError("Initial message history must begin at version zero.")
        elif not isinstance(parent, str) or not parent or parent == memory_id:
            raise ConversationError("A history commit has an invalid parent.")
        memory_event("Retrieve", "message_history_commit", version=value["version"],
                     sequence=value["sequence"], snapshot_id=value["snapshot_id"])
        return value

    @staticmethod
    def _transition(current: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
        if (current["sequence"] != parent["sequence"] + 1
                or current["version"] != parent["version"] + int(current["compacted"])):
            raise ConversationError("Message-history version or sequence continuity is broken.")

    def _snapshot(self, memory_id: str, commit: Mapping[str, Any]) -> HistorySnapshot:
        fragments: dict[int, list[dict[str, Any]]] = {}
        next_id = commit["first_memory_id"]
        seen: set[str] = set()
        last_position = (-1, -1)
        for index in range(commit["chunk_count"]):
            if not isinstance(next_id, str) or not next_id or next_id in seen:
                raise ConversationError("The message-history chunk chain is incomplete or cyclic.")
            seen.add(next_id)
            record = self.memory.get(next_id, run_id=self.chunk_run_id)
            metadata = record.get("metadata") or {}
            if (metadata.get("memory_kind") != "message_history_chunk"
                    or metadata.get("session_id") != self.settings.mem0_session_id
                    or metadata.get("ner_processed") is not True
                    or not isinstance(metadata.get("entities"), list)
                    or metadata.get("chunk_index") != index + 1
                    or any(metadata.get(key) != commit[key] for key in (
                        "version", "sequence", "snapshot_id", "content_sha256", "chunk_count",
                        "message_count", "compacted", "saved_at",
                    )) or metadata.get("source_audit_id") != commit["audit_id"]):
                raise ConversationError("A message-history chunk has inconsistent metadata.")
            try:
                chunk = json.loads(record["memory"])
                if (not isinstance(chunk, dict) or chunk.get("role") not in {"user", "assistant"}
                        or not isinstance(chunk.get("content"), str)
                        or ("name" in chunk and not isinstance(chunk["name"], str))
                        or any(type(chunk.get(key)) is not int or chunk[key] != metadata.get(key)
                               for key in ("message_index", "fragment_index", "fragment_count"))
                        or not 0 <= chunk["message_index"] < commit["message_count"]
                        or chunk["fragment_count"] < 1
                        or not 0 <= chunk["fragment_index"] < chunk["fragment_count"]):
                    raise ValueError("Invalid fragment fields")
            except (KeyError, TypeError, ValueError) as exc:
                raise ConversationError("A message-history fragment is invalid.") from exc
            position = (chunk["message_index"], chunk["fragment_index"])
            if position <= last_position:
                raise ConversationError("Message-history fragments are out of order.")
            last_position = position
            fragments.setdefault(chunk["message_index"], []).append(chunk)
            next_id = metadata.get("next_memory_id")
        if next_id is not None or sorted(fragments) != list(range(commit["message_count"])):
            raise ConversationError("The message-history chain has missing or extra messages.")
        messages: list[dict[str, str]] = []
        for message_index in range(commit["message_count"]):
            parts = fragments[message_index]
            if ([part["fragment_index"] for part in parts] != list(range(len(parts)))
                    or any(part["fragment_count"] != len(parts) for part in parts)
                    or len({part["role"] for part in parts}) != 1
                    or len({("name" in part, part.get("name")) for part in parts}) != 1):
                raise ConversationError("Message-history fragments are missing, reordered, or inconsistent.")
            messages.append({"role": parts[0]["role"], "content": "".join(part["content"] for part in parts),
                             **({"name": parts[0]["name"]} if "name" in parts[0] else {})})
        digest = hashlib.sha256(self._json(messages).encode("utf-8")).hexdigest()
        if digest != commit["content_sha256"]:
            raise ConversationError("Message history failed its content-hash check.")
        memory_event("Retrieve", "message_history_chunks", version=commit["version"],
                     sequence=commit["sequence"], chunk_count=commit["chunk_count"],
                     message_count=len(messages), verified=True)
        return HistorySnapshot(
            messages=messages, memory_id=memory_id,
            **{key: commit[key] for key in (
                "version", "sequence", "snapshot_id", "audit_id", "compacted", "saved_at",
                "content_sha256", "parent_memory_id",
            )},
        )

    def load_latest(self) -> HistorySnapshot | None:
        """Read the published snapshot afresh, including after a process restart."""
        memory_event("Retrieve", "latest_message_history", status="started")
        head = self._head()
        if head is None:
            memory_event("Retrieve", "latest_message_history", message_count=0,
                         reason="no_prior_history")
            return None
        pointer = head[1]
        commit = self._commit(pointer["memory_id"])
        if any(pointer.get(key) != commit[key] for key in ("version", "sequence", "snapshot_id")):
            raise ConversationError("The message-history head does not match its commit.")
        if commit["parent_memory_id"] is not None:
            self._transition(commit, self._commit(commit["parent_memory_id"]))
        snapshot = self._snapshot(pointer["memory_id"], commit)
        memory_event("Retrieve", "latest_message_history", version=snapshot.version,
                     sequence=snapshot.sequence, snapshot_id=snapshot.snapshot_id,
                     message_count=len(snapshot.messages))
        return snapshot

    def append(
        self, messages: Sequence[Mapping[str, str]], *, version: int,
        audit_id: str, compacted: bool = False,
    ) -> HistorySnapshot:
        """Append and verify a snapshot, then publish the navigation pointer last.

        The caller commits the full pre-compaction messages first, then calls
        this with a single summary and ``compacted=True``. Failed chunk or commit
        writes cannot change the published history. Orphan records are retained.
        """
        normalized = self._messages(messages)
        if type(version) is not int or version < 0 or type(compacted) is not bool:
            raise ConversationError("History version must be a nonnegative integer.")
        if not isinstance(audit_id, str) or not audit_id:
            raise ConversationError("History writes require a source audit ID.")
        if compacted and len(normalized) != 1:
            raise ConversationError("A compacted snapshot must contain one non-system message.")
        memory_event("Capture", "working_history_snapshot", status="started",
                     version=version, message_count=len(normalized), compacted=compacted)
        latest = self.load_latest()
        digest = hashlib.sha256(self._json(normalized).encode("utf-8")).hexdigest()
        if (latest is not None and latest.version == version and latest.audit_id == audit_id
                and latest.compacted == compacted and latest.content_sha256 == digest):
            memory_event("Capture", "working_history_snapshot", status="skipped",
                         reason="duplicate_verified", version=latest.version,
                         snapshot_id=latest.snapshot_id)
            return latest  # A retry after a lost response must not create another version.
        expected_version = (latest.version + int(compacted)) if latest is not None else 0
        if version != expected_version or (compacted and latest is None):
            raise ConversationError("History compaction must advance the latest committed version by one.")

        chunks = self.manager._conversation_chunks(normalized)
        for chunk in chunks:
            message = normalized[chunk["message_index"]]
            if "name" in message:
                chunk["name"] = message["name"]
        shared = {
            "version": version, "sequence": latest.sequence + 1 if latest else 1,
            "snapshot_id": uuid4().hex, "content_sha256": digest,
            "source_audit_id": audit_id, "saved_at": datetime.now(timezone.utc).isoformat(),
            "compacted": compacted,
        }
        next_id = None
        for index in range(len(chunks) - 1, -1, -1):
            chunk = chunks[index]
            raw = self._json(chunk)
            label = f"{chunk['role']}: {chunk['content']}"
            if "name" in chunk:
                label = f"{chunk['name']} ({chunk['role']}): {chunk['content']}"
            metadata = self._metadata(
                "message_history_chunk", label,
                **shared, chunk_index=index + 1, chunk_count=len(chunks),
                message_count=len(normalized), next_memory_id=next_id,
                **{key: chunk[key] for key in ("message_index", "fragment_index", "fragment_count")},
            )
            next_id = self.manager._created_id(self.memory.add(
                raw, run_id=self.chunk_run_id, metadata=metadata, infer=False,
            ))
            memory_event("Capture", "working_history_chunk", chunks=1,
                         chunk_index=index + 1, version=version, sequence=shared["sequence"],
                         snapshot_id=shared["snapshot_id"], stage="write_acknowledged")

        commit = {
            "schema": 1, "user_id": self.settings.mem0_user_id,
            "session_id": self.settings.mem0_session_id,
            **{key: shared[key] for key in (
                "version", "sequence", "snapshot_id", "content_sha256", "saved_at", "compacted",
            )},
            "audit_id": audit_id, "parent_memory_id": latest.memory_id if latest else None,
            "first_memory_id": next_id, "chunk_count": len(chunks), "message_count": len(normalized),
        }
        self._snapshot("unpublished", commit)  # No commit marker until every chunk verifies.
        raw = self._json(commit)
        commit_id = self.manager._created_id(self.memory.add(
            raw, run_id=self.run_id,
            metadata=self._metadata("message_history_commit", raw, **shared), infer=False,
        ))
        memory_event("Capture", "working_history_commit", commits=1, version=version,
                     sequence=shared["sequence"], snapshot_id=shared["snapshot_id"],
                     stage="write_acknowledged")
        if self._commit(commit_id) != commit:
            raise ConversationError("The immutable history commit failed its read-back check.")

        head = self._head()
        if ((latest is None and head is not None)
                or (latest is not None and (head is None or head[1]["memory_id"] != latest.memory_id))):
            raise ConversationError("Message history changed while saving; use one session writer.")
        pointer = {
            "schema": 1, "user_id": self.settings.mem0_user_id,
            "session_id": self.settings.mem0_session_id, "memory_id": commit_id,
            **{key: shared[key] for key in ("version", "sequence", "snapshot_id")},
        }
        raw = self._json(pointer)
        metadata = self._metadata("message_history_head", raw, **shared)
        if head is None:
            self.memory.add(raw, run_id=self.head_run_id, metadata=metadata, infer=False)
        else:
            self.memory.update(head[0], run_id=self.head_run_id, text=raw, metadata=metadata)
        memory_event("Capture" if head is None else "Update", "working_history_head",
                     version=version, sequence=shared["sequence"], snapshot_id=shared["snapshot_id"],
                     stage="write_acknowledged", older_snapshots_retained=True)
        published = self.load_latest()
        if published is None or published.memory_id != commit_id:
            raise ConversationError("The message-history navigation pointer could not be verified.")
        memory_event("Capture", "working_history_snapshot", version=published.version,
                     sequence=published.sequence, snapshot_id=published.snapshot_id,
                     message_count=len(published.messages), chunk_count=len(chunks),
                     compacted=published.compacted, verified=True)
        return published

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List committed snapshots newest first without a capped global listing."""
        memory_event("Retrieve", "message_history_versions", status="started")
        head = self._head()
        if head is None:
            memory_event("Retrieve", "message_history_versions", snapshot_count=0,
                         reason="no_prior_history")
            return []
        memory_id = head[1]["memory_id"]
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        child = None
        while memory_id is not None:
            if memory_id in seen:
                raise ConversationError("The message-history commit chain contains a cycle.")
            seen.add(memory_id)
            commit = self._commit(memory_id)
            if child is not None:
                self._transition(child, commit)
            elif any(head[1].get(key) != commit[key] for key in ("version", "sequence", "snapshot_id")):
                raise ConversationError("The message-history head does not match its commit.")
            result.append({"memory_id": memory_id, **{key: commit[key] for key in (
                "version", "sequence", "snapshot_id", "audit_id", "compacted", "saved_at",
                "content_sha256", "parent_memory_id", "message_count", "chunk_count",
            )}})
            child, memory_id = commit, commit["parent_memory_id"]
        memory_event("Retrieve", "message_history_versions", snapshot_count=len(result),
                     latest_version=result[0]["version"])
        return result
