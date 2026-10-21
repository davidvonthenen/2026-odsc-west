"""Official Mem0 SDK with a narrow transport for the v2.2.1 OSS server.

MemoryClient speaks the hosted API contract, even when ``host`` is overridden.
OSSServerTransport translates only the operations used here. Memory extraction,
embeddings, storage and history remain in the remote Mem0 container.
"""
from __future__ import annotations

import json
import os
import re
from importlib.metadata import version
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from .config import Settings
from .logging import memory_event

SDK_VERSION = "2.2.1"
OSS_LIST_LIMIT = 1000
MEMORY_DOMAIN = "technology_news"


class Mem0ServiceError(RuntimeError):
    """Raised when the Mem0 SDK cannot complete a memory operation."""


def _server_url(value: str) -> str:
    """Require an explicit server origin rather than a hosted-service fallback."""
    value = value.strip().rstrip("/")
    url = urlsplit(value)
    if (url.scheme not in {"http", "https"} or not url.hostname
            or url.username or url.password or url.path or url.query or url.fragment):
        raise Mem0ServiceError(
            "MEM0_API_URL must be a server origin, such as http://127.0.0.1:8888, "
            "without credentials, /v1, a path, or query parameters."
        )
    if url.hostname.lower() == "api.mem0.ai":
        raise Mem0ServiceError("MEM0_API_URL must point to your self-hosted Mem0 server.")
    # Validate an explicitly supplied port before constructing the HTTP client.
    try:
        url.port
    except ValueError as exc:
        raise Mem0ServiceError("MEM0_API_URL has an invalid port.") from exc
    return value


class OSSServerTransport(httpx.BaseTransport):
    """Translate SDK requests to the unchanged v2.2.1 self-hosted REST API.

    This is an application compatibility adapter, not an upstream Mem0 feature.
    Unsupported operations fail instead of forwarding hosted-only API requests.
    The optional inner transport is for offline HTTP-contract tests.
    """

    def __init__(
        self, api_key: str, *, inner: httpx.BaseTransport | None = None
    ) -> None:
        self.api_key = api_key.strip()
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0, trust_env=False)
        self.server_config: Dict[str, Any] = {}

    @staticmethod
    def _only(payload: Mapping[str, Any], allowed: set[str]) -> None:
        unsupported = set(payload) - allowed
        if unsupported:
            raise Mem0ServiceError(
                f"Unsupported OSS request fields: {', '.join(sorted(unsupported))}. "
                f"Use mem0ai=={SDK_VERSION} and the supplied application adapter."
            )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        route = (request.method, request.url.path)
        payload = json.loads(request.content) if request.content else {}
        if not isinstance(payload, dict):
            raise Mem0ServiceError("Expected a JSON object in the SDK request.")
        method, path = route
        params: Dict[str, Any] = {}
        body: Dict[str, Any] | None = payload if request.content else None

        if route == ("GET", "/v1/ping/"):
            path = "/configure"  # Authenticate before translating the ping response.
        elif route == ("POST", "/v3/memories/add/"):
            self._only(payload, {"messages", "user_id", "run_id", "metadata", "infer"})
            path = "/memories"
        elif route == ("POST", "/v3/memories/search/"):
            self._only(payload, {"query", "filters", "top_k", "threshold"})
            path = "/search"
        elif route == ("POST", "/v3/memories/"):
            self._only(payload, {"filters"})
            if request.url.query:
                raise Mem0ServiceError("The OSS server has no list pagination endpoint.")
            filters = payload.get("filters") or {}
            self._only(filters, {"user_id", "run_id"})
            if not filters.get("user_id") or not filters.get("run_id"):
                raise Mem0ServiceError("Listing requires both user_id and run_id.")
            method, path, body = "GET", "/memories", None
            params = {**filters, "top_k": OSS_LIST_LIMIT}
        elif method in {"GET", "PUT"} and re.fullmatch(
            r"/v1/memories/[A-Za-z0-9_-]+/", path
        ):
            path = path.replace("/v1/memories/", "/memories/", 1).rstrip("/")
            if method == "PUT":
                self._only(payload, {"text", "metadata"})
        else:
            raise Mem0ServiceError(f"Unsupported OSS SDK operation: {method} {path}")

        headers = httpx.Headers(request.headers)
        for name in ("authorization", "x-api-key", "content-length", "content-type", "host"):
            headers.pop(name, None)
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        translated = httpx.Request(
            method,
            request.url.copy_with(path=path, query=b""),
            params=params,
            json=body,
            headers=headers,
            extensions=request.extensions,
        )
        response = self.inner.handle_request(translated)
        # Buffer before returning so SDK errors retain the original SDK request,
        # while the actual wire request uses the OSS path and authentication.
        try:
            content = response.read()
        finally:
            response.close()
        if route == ("GET", "/v1/ping/") and 200 <= response.status_code < 300:
            try:
                config = json.loads(content)
            except (ValueError, TypeError) as exc:
                raise Mem0ServiceError("Mem0 /configure did not return JSON.") from exc
            if not isinstance(config, dict) or not isinstance(config.get("vector_store"), dict):
                raise Mem0ServiceError("The endpoint did not return a Mem0 server configuration.")
            self.server_config = config
            # The SDK eagerly constructs Project from its ping response. The OSS
            # server has no Platform tenancy; these are adapter-only labels, not
            # MEMORY_USER_ID / MEMORY_SESSION_ID or server credentials. Emit them
            # only after the real authenticated /configure request is validated.
            content = json.dumps({
                "org_id": "self-hosted",
                "project_id": "self-hosted",
                "user_email": None,
            }).encode("utf-8")
        response_headers = httpx.Headers(response.headers)
        # httpx may already have decoded compression when response.read() ran.
        for name in ("content-encoding", "content-length", "transfer-encoding"):
            response_headers.pop(name, None)
        return httpx.Response(response.status_code, headers=response_headers, content=content)

    def close(self) -> None:
        self.inner.close()


class Mem0Client:
    """Persist and retrieve episodic records through the official remote SDK."""

    def __init__(self, settings: Settings, memory: Any | None = None) -> None:
        self.settings = settings
        self._http: httpx.Client | None = None
        self._transport: OSSServerTransport | None = None
        self._closed = False
        if memory is not None:  # Dependency injection for tests, not a runtime fallback.
            self.memory = memory
            return

        host = _server_url(settings.mem0_api_url)
        if not settings.mem0_user_id.strip() or not settings.mem0_session_id.strip():
            raise Mem0ServiceError("MEMORY_USER_ID and MEMORY_SESSION_ID must be nonempty.")
        if settings.mem0_timeout <= 0:
            raise Mem0ServiceError("MEM0_TIMEOUT must be greater than zero.")
        try:
            installed = version("mem0ai")
            if installed != SDK_VERSION:
                raise Mem0ServiceError(
                    f"This adapter requires mem0ai=={SDK_VERSION}; installed version is {installed}."
                )
            # Set before importing the SDK. This does not create a local Memory engine.
            os.environ.setdefault("MEM0_TELEMETRY", "false")
            from mem0 import MemoryClient

            self._transport = OSSServerTransport(settings.mem0_api_key)
            self._http = httpx.Client(
                base_url=host,
                transport=self._transport,
                timeout=settings.mem0_timeout,
                follow_redirects=False,
                trust_env=False,
            )
            self.memory = MemoryClient(
                host=host,
                # SDK requires a nonempty value. The placeholder is never sent to
                # the server: only an actual configured key becomes X-API-Key.
                api_key=settings.mem0_api_key or "local-auth-disabled",
                client=self._http,
            )
        except Exception as exc:
            if self._http is not None:
                self._http.close()
            raise Mem0ServiceError(
                f"Mem0 SDK connection to {host} failed: {exc}. "
                "Install the pinned requirements and use the API key printed by "
                "server/scripts/seed.sh or created in the local dashboard. "
                "An empty MEM0_API_KEY works only when the server has AUTH_DISABLED=true."
            ) from exc

    @property
    def backend_provider(self) -> str:
        """Report the backend returned by the authenticated server configuration."""
        config = self._transport.server_config if self._transport is not None else {}
        return str((config.get("vector_store") or {}).get("provider") or "unknown")

    @staticmethod
    def _results(response: Any) -> List[Dict[str, Any]]:
        """Normalize results without converting failed/malformed responses to emptiness."""
        if isinstance(response, dict):
            response = response.get("results")
        if not isinstance(response, list) or any(not isinstance(item, dict) for item in response):
            raise Mem0ServiceError("Mem0 returned an invalid results payload.")
        return [dict(item) for item in response]

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _messages(content: str | Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
        """Preserve message contents, including whitespace within JSON fragments."""
        if isinstance(content, str):
            if not content.strip():
                raise ValueError("Memory content must not be empty")
            return [{"role": "user", "content": content}]
        messages: List[Dict[str, str]] = []
        for item in content:
            role, text = str(item.get("role") or ""), str(item.get("content") or "")
            if not role or not text:
                raise ValueError("Each memory message requires role and content")
            messages.append({"role": role, "content": text})
        if not messages:
            raise ValueError("Memory messages must not be empty")
        return messages

    def _call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        aspect = "Capture" if operation == "add" else "Update" if operation == "update" else "Retrieve"
        if self._closed:
            memory_event(aspect, f"mem0_{operation}", status="failed", reason="client_closed")
            raise Mem0ServiceError("Mem0 client is closed.")
        try:
            return getattr(self.memory, operation)(*args, **kwargs)
        except Exception as exc:
            memory_event(aspect, f"mem0_{operation}", status="failed", error_type=type(exc).__name__)
            raise Mem0ServiceError(f"Mem0 SDK {operation} failed: {exc}") from exc

    def _scope(self, record: Dict[str, Any], run_id: str | None = None) -> Dict[str, Any]:
        """Fail closed on a response for another user or conversation run."""
        metadata = dict(record.get("metadata") or {})
        user = record.get("user_id", metadata.get("user_id"))
        run = record.get("run_id", metadata.get("run_id"))
        if user != self.settings.mem0_user_id or (run_id is not None and run != run_id):
            raise Mem0ServiceError("Mem0 returned a record outside the requested user/run scope.")
        # OSS promotes identity out of metadata. Retain the application's read shape.
        if metadata.get("memory_domain") != MEMORY_DOMAIN:
            raise Mem0ServiceError("Mem0 returned a record outside the News Agent memory domain.")
        metadata["user_id"] = user
        record["metadata"] = metadata
        return record

    def _filters(self, run_id: str | None, metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
        filters = dict(metadata or {})
        if any(key in filters for key in ("user_id", "run_id", "agent_id", "memory_domain", "AND", "OR", "NOT")):
            raise ValueError("metadata_filters cannot override identity or introduce logical scope filters.")
        filters["memory_domain"] = MEMORY_DOMAIN
        filters["user_id"] = self.settings.mem0_user_id
        if run_id is not None:
            if not run_id:
                raise ValueError("run_id must be nonempty")
            filters["run_id"] = run_id
        return filters

    def add(
        self, content: str | Sequence[Mapping[str, str]], *, run_id: str,
        metadata: Mapping[str, Any], infer: bool,
    ) -> Any:
        """Add raw or inferred memory without selecting a client-side database."""
        if not run_id:
            raise ValueError("run_id must be nonempty")
        merged = dict(metadata)
        if merged.get("ner_processed") is not True or not isinstance(merged.get("entities"), list):
            raise Mem0ServiceError("Every memory write requires successful NER metadata labeling.")
        merged["memory_domain"] = MEMORY_DOMAIN
        # v2.2.1 requires identity as top-level add parameters, not metadata.
        for key in ("user_id", "run_id", "agent_id", "actor_id"):
            merged.pop(key, None)
        messages = self._messages(content)
        response = self._call(
            "add", messages, user_id=self.settings.mem0_user_id,
            run_id=run_id, metadata=merged, infer=bool(infer),
        )
        results = self._results(response)
        if not infer and (len(results) != len(messages) or any(not item.get("id") for item in results)):
            raise Mem0ServiceError("Mem0 did not acknowledge every raw memory write.")
        memory_event("Capture", "mem0_add", memory_kind=merged.get("memory_kind"), infer=bool(infer),
                     **({"inference_results": len(results)} if infer else {"records": len(results)}))
        if infer:
            # Report only backend changes the server explicitly acknowledged.
            # One inferred-turn submission need not create one new memory.
            for event, aspect in (("ADD", "Capture"), ("UPDATE", "Update"),
                                  ("DELETE", "Forget"), ("NONE", "Capture")):
                count = sum(item.get("event") == event for item in results)
                if count:
                    memory_event(aspect, f"mem0_inferred_{event.lower()}", records=count,
                                 status="skipped" if event == "NONE" else "completed")
        return response

    def get(self, memory_id: str, *, run_id: str) -> Dict[str, Any]:
        """Read one exact record, used for lossless checkpoint traversal."""
        response = self._call("get", memory_id)
        if not isinstance(response, dict) or response.get("id") != memory_id:
            raise Mem0ServiceError("Mem0 returned an invalid single-memory payload.")
        record = self._scope(dict(response), run_id)
        memory_event("Retrieve", "mem0_get", memory_id=memory_id,
                     memory_kind=record["metadata"].get("memory_kind"), records=1)
        return record

    def update(self, memory_id: str, *, run_id: str, text: str, metadata: Mapping[str, Any]) -> Any:
        """Update only an already-verified record in this user's run."""
        record = self.get(memory_id, run_id=run_id)
        if metadata.get("ner_processed") is not True or not isinstance(metadata.get("entities"), list):
            raise Mem0ServiceError("Memory updates require successful NER metadata labeling.")
        merged = {**record["metadata"], **metadata, "memory_domain": MEMORY_DOMAIN}
        for key in ("user_id", "run_id", "agent_id", "actor_id"):
            merged.pop(key, None)
        response = self._call("update", memory_id, text=text, metadata=merged)
        memory_event("Update", "mem0_update", memory_id=memory_id,
                     memory_kind=merged.get("memory_kind"))
        return response

    def search(
        self, query: str, *, run_id: str | None = None,
        metadata_filters: Mapping[str, Any] | None = None, top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        response = self._call(
            "search", query, filters=self._filters(run_id, metadata_filters),
            top_k=int(top_k if top_k is not None else self.settings.mem0_top_k),
            threshold=self.settings.mem0_search_threshold,
        )
        results = [self._scope(item, run_id) for item in self._results(response)]
        if metadata_filters and any(
            any(item["metadata"].get(key) != value for key, value in metadata_filters.items())
            for item in results
        ):
            raise Mem0ServiceError("Mem0 returned a record outside the requested metadata scope.")
        memory_event("Retrieve", "mem0_search", records=len(results),
                     memory_kind=(metadata_filters or {}).get("memory_kind"))
        return results

    def search_runs(self, query: str, run_ids: Iterable[str], *, top_k: int | None = None) -> List[Dict[str, Any]]:
        limit = int(top_k if top_k is not None else self.settings.mem0_top_k)
        combined: Dict[str, Dict[str, Any]] = {}
        for run_id in run_ids:
            for result in self.search(query, run_id=run_id, top_k=limit):
                metadata = result.get("metadata") or {}
                key = str(result.get("id") or metadata.get("event_id") or f"{run_id}:{result.get('memory', '')}")
                current = combined.get(key)
                if current is None or self._score(result.get("score")) > self._score(current.get("score")):
                    combined[key] = result
        return sorted(combined.values(), key=lambda item: self._score(item.get("score")), reverse=True)[:limit]

    def get_all(
        self, *, run_id: str, metadata_filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Read a small exact scope; never pretend the OSS cap is pagination.

        Used for scoped inventories, including navigation heads. Snapshot chunks
        and message-history commits use exact IDs instead of a full listing.
        """
        self._filters(run_id, metadata_filters)  # Validate before any request.
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must not be negative")
        response = self._call("get_all", filters={
            "user_id": self.settings.mem0_user_id, "run_id": run_id,
        })
        results = [self._scope(item, run_id) for item in self._results(response)]
        if len(results) >= OSS_LIST_LIMIT:
            raise Mem0ServiceError(
                f"OSS listing for run {run_id!r} reached its {OSS_LIST_LIMIT}-record cap. "
                "Refusing to load a potentially truncated scoped inventory. "
                "Snapshot and history traversal use exact memory IDs."
            )
        if metadata_filters:
            results = [item for item in results if all(
                item["metadata"].get(key) == value for key, value in metadata_filters.items()
            )]
        selected = results if top_k is None else results[:top_k]
        memory_event("Retrieve", "mem0_get_all", records=len(selected))
        return selected

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._http is not None:
            self._http.close()
        else:
            close = getattr(self.memory, "close", None)
            if callable(close):
                close()


__all__ = ["Mem0Client", "Mem0ServiceError"]
