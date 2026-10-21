"""Reset selected sessions for one user in a self-hosted Mem0 v2.2.1 instance.

Edit the constants below, stop all writers for the selected user/sessions, then run:
    python reset_memory.py

All application logic is contained in this file. It does not import common/ or any
other project module.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
import json
import re
import sys
from typing import Any, Iterator, Sequence
from urllib.parse import urlsplit

import httpx

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SDK_VERSION = "2.2.1"

MEM0_API_URL = "http://127.0.0.1:8888"
# Set this to the local Mem0 instance API key when authentication is enabled.
# Leave it blank only when server authentication is disabled.
MEM0_API_KEY = ""
MEM0_TIMEOUT = 30.0

# The OSS server does not have Mem0 Platform tenancy. These constants satisfy the
# official SDK's eager Project initialization and are used only by this adapter.
ORG_ID = "self-hosted"
PROJECT_ID = "self-hosted"

MEMORY_USER_ID = "user-123"
MEMORY_SESSION_ID = [
    "123456",
]

BATCH_SIZE = 1000
MEMORY_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
RUN_SUFFIXES = ("complete", "synthesized", "internet", "complete-head")


class ResetError(RuntimeError):
    """A reset could not finish safely; previously completed deletions remain."""


def server_url(value: str) -> str:
    """Validate the self-hosted origin before sending any credentials."""
    value = value.strip().rstrip("/")
    try:
        url = urlsplit(value)
        port = url.port
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.path
            or url.query
            or url.fragment
            or url.hostname.lower().rstrip(".") == "api.mem0.ai"
            or port == 0
        ):
            raise ValueError
    except ValueError as exc:
        raise ResetError(
            "MEM0_API_URL must be your self-hosted server origin, for example "
            "http://127.0.0.1:8888, without credentials, API paths, query "
            "parameters, or fragments."
        ) from exc
    return value


@dataclass(frozen=True)
class ResetSettings:
    """Configuration required by this standalone utility."""

    api_url: str
    api_key: str = field(repr=False)
    timeout: float = 300.0
    user_id: str = ""
    session_ids: tuple[str, ...] = ()

    def run_ids_for_session(self, session_id: str) -> tuple[str, ...]:
        """Keep the recovery checkpoint until the other runs have been drained."""
        if session_id not in self.session_ids:
            raise ResetError(f"Unknown configured session: {session_id!r}")
        return tuple(f"{session_id}-{suffix}" for suffix in RUN_SUFFIXES)

    @property
    def allowed_run_ids(self) -> frozenset[str]:
        return frozenset(
            run_id
            for session_id in self.session_ids
            for run_id in self.run_ids_for_session(session_id)
        )


def build_settings() -> ResetSettings:
    """Build configuration exclusively from the constants at the top of this file."""
    user_id = str(MEMORY_USER_ID).strip()
    if not user_id:
        raise ResetError("MEMORY_USER_ID must be a nonempty string.")

    if not isinstance(MEMORY_SESSION_ID, (list, tuple)):
        raise ResetError("MEMORY_SESSION_ID must be a list or tuple of session IDs.")

    session_ids = tuple(str(session_id).strip() for session_id in MEMORY_SESSION_ID)
    if not session_ids or any(not session_id for session_id in session_ids):
        raise ResetError("MEMORY_SESSION_ID must contain at least one nonempty session ID.")
    if len(session_ids) != len(set(session_ids)):
        raise ResetError("MEMORY_SESSION_ID contains duplicate session IDs.")

    try:
        timeout = float(MEM0_TIMEOUT)
    except (TypeError, ValueError) as exc:
        raise ResetError("MEM0_TIMEOUT must be a number greater than zero.") from exc
    if timeout <= 0:
        raise ResetError("MEM0_TIMEOUT must be greater than zero.")

    return ResetSettings(
        api_url=server_url(str(MEM0_API_URL)),
        api_key=str(MEM0_API_KEY).strip(),
        timeout=timeout,
        user_id=user_id,
        session_ids=session_ids,
    )


class ResetTransport(httpx.BaseTransport):
    """Map official SDK read/delete routes to the self-hosted v2.2.1 OSS API.

    Only authentication, exact-run listing, single-ID reads, and single-ID deletes
    are allowed. No bulk deletion, search, writes, or global reset is forwarded.
    """

    def __init__(
        self,
        settings: ResetSettings,
        *,
        inner: httpx.BaseTransport | None = None,
    ) -> None:
        if not settings.user_id.strip() or not settings.session_ids:
            raise ResetError("A user and at least one session ID are required.")
        self.settings = settings
        self.origin = httpx.URL(server_url(settings.api_url))
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0, trust_env=False)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if (request.url.scheme, request.url.host, request.url.port) != (
            self.origin.scheme,
            self.origin.host,
            self.origin.port,
        ):
            raise ResetError("Refusing to send Mem0 credentials to a different origin.")
        if request.url.query:
            raise ResetError("Unexpected SDK query parameters; refusing to broaden the operation.")

        payload = json.loads(request.content) if request.content else {}
        if not isinstance(payload, dict):
            raise ResetError("Expected an SDK JSON object.")

        route = request.method, request.url.path
        method, path = route
        params: dict[str, Any] = {}

        if route == ("GET", "/v1/ping/") and not payload:
            path = "/configure"
        elif route == ("POST", "/v3/memories/"):
            filters = payload.get("filters")
            if (
                set(payload) != {"filters", "show_expired"}
                or payload["show_expired"] is not True
                or not isinstance(filters, dict)
                or set(filters) != {"user_id", "run_id"}
                or filters["user_id"] != self.settings.user_id
                or filters["run_id"] not in self.settings.allowed_run_ids
            ):
                raise ResetError(
                    "Listing must specify the exact configured user and one allowed session run."
                )
            method, path = "GET", "/memories"
            params = {**filters, "top_k": BATCH_SIZE, "show_expired": "true"}
        elif (
            method in {"GET", "DELETE"}
            and not payload
            and re.fullmatch(r"/v1/memories/[A-Za-z0-9_-]+/", path)
        ):
            path = path.replace("/v1/memories/", "/memories/", 1).rstrip("/")
        else:
            raise ResetError(f"Unsupported reset operation: {method} {path}")

        # Never forward the SDK's Platform auth header. The self-hosted server
        # accepts its local instance key as X-API-Key.
        headers = {"Accept": "application/json"}
        if self.settings.api_key:
            headers["X-API-Key"] = self.settings.api_key

        translated = httpx.Request(
            method,
            self.origin.copy_with(path=path),
            params=params,
            headers=headers,
            extensions=request.extensions,
        )
        response = self.inner.handle_request(translated)
        try:
            content = response.read()
        finally:
            response.close()

        if route == ("GET", "/v1/ping/") and 200 <= response.status_code < 300:
            config = json.loads(content)
            if not isinstance(config, dict) or not isinstance(config.get("vector_store"), dict):
                raise ResetError("The endpoint did not return a Mem0 server configuration.")

            # MemoryClient eagerly constructs Project from /v1/ping/. The OSS
            # server has no Platform tenancy, so expose the configured adapter-only
            # labels after the real authenticated /configure call succeeds.
            content = json.dumps(
                {
                    "org_id": ORG_ID,
                    "project_id": PROJECT_ID,
                    "user_email": None,
                }
            ).encode("utf-8")

        headers_out = httpx.Headers(response.headers)
        for name in ("content-length", "content-encoding", "transfer-encoding"):
            headers_out.pop(name, None)
        return httpx.Response(response.status_code, headers=headers_out, content=content)

    def close(self) -> None:
        self.inner.close()


@contextmanager
def connect(settings: ResetSettings) -> Iterator[Any]:
    """Use the official remote SDK; never instantiate a local Memory engine."""
    installed = version("mem0ai")
    if installed != SDK_VERSION:
        raise ResetError(f"Install mem0ai=={SDK_VERSION}; found {installed}.")

    from mem0 import MemoryClient

    with httpx.Client(
        base_url=settings.api_url,
        transport=ResetTransport(settings),
        timeout=settings.timeout,
        follow_redirects=False,
        trust_env=False,
    ) as http:
        # MemoryClient requires a nonempty API key even when the OSS server has
        # authentication disabled. The placeholder is consumed only by the SDK;
        # ResetTransport never forwards it to the OSS server.
        yield MemoryClient(
            host=settings.api_url,
            api_key=settings.api_key or "local-auth-disabled",
            client=http,
        )


def validate_record(
    record: Any,
    settings: ResetSettings,
    session_id: str,
    run_id: str,
) -> str:
    """Reject malformed or foreign records before deletion."""
    if session_id not in settings.session_ids:
        raise ResetError(f"Unknown configured session: {session_id!r}")
    if run_id not in settings.run_ids_for_session(session_id):
        raise ResetError("Deletion is restricted to this application's exact session runs.")
    if not isinstance(record, dict):
        raise ResetError("Mem0 returned a malformed record.")

    metadata = record.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ResetError("Mem0 returned malformed metadata.")

    for key, expected in (("user_id", settings.user_id), ("run_id", run_id)):
        actual = record.get(key, metadata.get(key))
        if actual != expected or (key in metadata and metadata[key] != expected):
            raise ResetError("Mem0 returned a record outside the requested user/run scope.")

    if "session_id" in metadata and metadata["session_id"] != session_id:
        raise ResetError("Mem0 returned conflicting session metadata.")

    memory_id = record.get("id")
    if not isinstance(memory_id, str) or not MEMORY_ID_PATTERN.fullmatch(memory_id):
        raise ResetError("Mem0 returned an invalid memory ID.")
    return memory_id


def list_batch(
    client: Any,
    settings: ResetSettings,
    session_id: str,
    run_id: str,
) -> list[str]:
    """Retrieve and validate an entire batch before deleting any records."""
    if run_id not in settings.run_ids_for_session(session_id):
        raise ResetError("Deletion is restricted to this application's exact session runs.")

    response = client.get_all(
        filters={"user_id": settings.user_id, "run_id": run_id},
        show_expired=True,
    )
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise ResetError("Mem0 returned an invalid results payload.")
    if response.get("next"):
        raise ResetError("Unexpected paginated response from the OSS server.")

    ids = [
        validate_record(row, settings, session_id, run_id)
        for row in response["results"]
    ]
    if len(ids) != len(set(ids)):
        raise ResetError("Mem0 returned duplicate IDs in a deletion batch.")
    return ids


def delete_run(
    client: Any,
    settings: ResetSettings,
    session_id: str,
    run_id: str,
) -> int:
    """Drain the capped listing, including expired and orphaned snapshot records."""
    deleted: set[str] = set()
    try:
        while True:
            ids = list_batch(client, settings, session_id, run_id)
            if not ids:
                return len(deleted)
            if deleted.intersection(ids):
                raise ResetError(
                    "Deleted records are still listed; refusing to loop or report success."
                )

            for memory_id in ids:
                # Recheck ownership immediately before each single-record SDK deletion.
                record = client.get(memory_id)
                if validate_record(record, settings, session_id, run_id) != memory_id:
                    raise ResetError("Mem0 returned the wrong record for the requested ID.")

                response = client.delete(memory_id)
                if (
                    not isinstance(response, dict)
                    or response.get("message") != "Memory deleted successfully"
                ):
                    raise ResetError("Mem0 did not acknowledge the memory deletion.")
                deleted.add(memory_id)
    except Exception as exc:
        raise ResetError(
            f"Run {run_id!r} stopped after {len(deleted)} acknowledged deletions: {exc}. "
            "Some records may already be deleted; keep writers stopped and rerun to finish."
        ) from exc


def reset_session(client: Any, settings: ResetSettings, session_id: str) -> int:
    """Delete one configured session's four application runs and verify emptiness."""
    total = 0
    for run_id in settings.run_ids_for_session(session_id):
        count = delete_run(client, settings, session_id, run_id)
        total += count
        print(f"Deleted {count} records from {run_id!r}.", flush=True)

    for run_id in settings.run_ids_for_session(session_id):
        if list_batch(client, settings, session_id, run_id):
            raise ResetError(
                f"Records appeared during final verification for session {session_id!r}. "
                "Stop writers and rerun."
            )
    return total


def reset_sessions(client: Any, settings: ResetSettings) -> int:
    """Reset every configured session for MEMORY_USER_ID."""
    total = 0
    for session_id in settings.session_ids:
        print(f"\nResetting session {session_id!r} for user {settings.user_id!r}...", flush=True)
        count = reset_session(client, settings, session_id)
        total += count
        print(f"Session {session_id!r} reset complete: {count} records deleted.", flush=True)
    return total


def main(argv: Sequence[str] | None = None) -> int:
    """Display the exact scope and require confirmation before connecting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    try:
        settings = build_settings()

        print(f"Mem0 endpoint: {settings.api_url}")
        print(f"SDK org_id: {ORG_ID!r}")
        print(f"SDK project_id: {PROJECT_ID!r}")
        print(f"User: {settings.user_id!r}")
        print("Sessions and runs to delete:")
        for session_id in settings.session_ids:
            print(f"  Session {session_id!r}")
            for run_id in settings.run_ids_for_session(session_id):
                print(f"    {run_id!r}")

        print("Stop all query.py processes writing this user or these sessions before continuing.")
        print("This deletes application memory records, not server audit history or backups.")
        if input("Type DELETE to reset all listed sessions, or press Enter to cancel: ").strip() != "DELETE":
            print("Cancelled. No memory was deleted.")
            return 0

        with connect(settings) as client:
            total = reset_sessions(client, settings)

        print(
            f"\nReset complete: deleted {total} records across "
            f"{len(settings.session_ids)} session(s) for user {settings.user_id!r}."
        )
        print("Unlisted sessions for this user were not modified.")
        return 0
    except (EOFError, KeyboardInterrupt):
        print(
            "\nReset interrupted. Any completed deletions remain; rerun to finish.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"Memory reset failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
