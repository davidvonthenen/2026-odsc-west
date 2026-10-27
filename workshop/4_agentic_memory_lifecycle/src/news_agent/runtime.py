"""Run one async agent on a persistent event loop behind synchronous Flask views."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
import threading
from typing import Any, Callable


class RuntimeClosedError(RuntimeError):
    """New work was submitted after graceful shutdown began."""


class AgentRuntime:
    """Share async clients and session locks across WSGI request threads.

    Construction is lazy, so importing an app or checking its health does not
    start a thread or connect to services. One instance belongs to one process.
    """

    def __init__(self, agent_factory: Callable[[], Any]) -> None:
        self._factory = agent_factory
        self._agent: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._pending: set[Future] = set()
        self._closed = False

    def _run_loop(self) -> None:
        loop = self._loop
        assert loop is not None
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                loop.close()

    async def _invoke(self, query: str, session_id: str, **kwargs: Any) -> dict[str, Any]:
        if self._agent is None:
            # The factory and persistent async clients run on this loop only.
            self._agent = self._factory()
        return await self._agent.ainvoke(query, session_id, **kwargs)

    def invoke(self, query: str, session_id: str, **kwargs: Any) -> dict[str, Any]:
        """Wait for the complete governed turn, including its memory persistence."""
        with self._lock:
            if self._closed:
                raise RuntimeClosedError("The agent runtime is shutting down")
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._run_loop, name="news-agent-loop", daemon=True,
                )
                self._thread.start()
            future = asyncio.run_coroutine_threadsafe(
                self._invoke(query, session_id, **kwargs), self._loop,
            )
            self._pending.add(future)
        try:
            return future.result()
        finally:
            with self._lock:
                self._pending.discard(future)

    async def _cleanup(self) -> None:
        if self._agent is not None:
            close = getattr(self._agent, "aclose", None)
            if close is not None:
                await close()

    def close(self) -> None:
        """Reject new turns, drain accepted turns, close clients, then stop the loop.

        Call from the WSGI/main thread, never from the agent's event-loop thread.
        Draining avoids releasing a session lock while a memory write is active.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("Runtime shutdown must run outside the agent loop")
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                pending = tuple(self._pending)
                loop = self._loop
                thread = self._thread
            for future in pending:
                try:
                    future.result()
                except Exception:
                    # The owning HTTP request handles its exception.
                    pass
            if loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(self._cleanup(), loop).result()
                finally:
                    loop.call_soon_threadsafe(loop.stop)
                    assert thread is not None
                    thread.join()
