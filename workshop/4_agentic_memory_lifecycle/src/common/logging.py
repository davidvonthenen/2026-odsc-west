"""Logging utilities."""
from __future__ import annotations

import logging
import json
import threading
from contextvars import ContextVar
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with concise console formatting."""

    logger = logging.getLogger(name if name else __name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


MEMORY_OPERATIONS = ("Capture", "Consolidate", "Retrieve", "Update", "Forget")
MEMORY_LOGGER = get_logger("memory.lifecycle")


class MemoryLifecycleLog:
    """Console event totals for one HTTP response, shared with its async workers."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.counts = {
            operation: {"completed": 0, "skipped": 0, "failed": 0}
            for operation in MEMORY_OPERATIONS
        }
        self._lock = threading.Lock()

    def summary(self, *, http_status: int, outcome: str, **details: object) -> None:
        """Report events, not a sum of unlike units such as chunks and messages."""
        with self._lock:
            counts = {operation: dict(totals) for operation, totals in self.counts.items()}
        MEMORY_LOGGER.info(
            "memory_summary request_id=%s http_status=%s outcome=%s operations=%s details=%s",
            self.request_id, http_status, outcome,
            json.dumps(counts, separators=(",", ":")),
            json.dumps(details, ensure_ascii=True, separators=(",", ":")),
        )


# Context follows run_coroutine_threadsafe and asyncio.to_thread. A mutable,
# request-owned collector lets those copied contexts contribute to one summary.
MEMORY_LIFECYCLE: ContextVar[MemoryLifecycleLog | None] = ContextVar(
    "memory_lifecycle", default=None,
)


def memory_event(
    operation: str, action: str, *, status: str = "completed", **details: object,
) -> None:
    """Print one lifecycle event immediately; count only terminal outcomes.

    Call sites supply counts, identifiers and error types, never memory content.
    JSON escaping keeps caller-controlled identifiers on a single console line.
    """
    tracker = MEMORY_LIFECYCLE.get()
    if tracker is not None and status in tracker.counts[operation]:
        with tracker._lock:
            tracker.counts[operation][status] += 1
    MEMORY_LOGGER.info(
        "memory_operation request_id=%s operation=%s action=%s status=%s details=%s",
        tracker.request_id if tracker is not None else "-", operation, action, status,
        json.dumps(details, ensure_ascii=True, separators=(",", ":")),
    )


__all__ = ["get_logger", "MemoryLifecycleLog", "MEMORY_LIFECYCLE", "memory_event"]
