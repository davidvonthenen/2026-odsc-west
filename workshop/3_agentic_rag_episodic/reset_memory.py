"""Reset the Agentic News Agent's selected sessions in self-hosted Mem0 2.2.1.

Setup on the host where start_mem0 was run:
    python -m pip install 'mem0ai==2.2.1' 'httpx>=0.28,<1' 'requests>=2.32,<3'

Edit the user/session constants below, then run:
    python reset_memory_new.py

Use the same MEMORY_USER_ID as the News Agent (or its trusted REMOTE_USER).
This utility applies the agent's news-agent: prefix. MEMORY_SESSION_ID is a list
of the exact sessions to reset; it corresponds to the REST request's user field.
Configuration comes from this file's constants, not the project's .env file.

Stop the News Agent server(s) and let in-flight writes finish before confirming.
Keep the Mem0 server running. The utility deletes the complete, synthesized,
internet and complete-head runs for the selected user/sessions. It then removes
Mem0's recent extraction messages for the same exact scopes and verifies both.
Other sessions remain available to the agent's cross-session topic recall.

Storage: pgvector holds vectors and memory payload/metadata when configured on
the server. Stock Mem0 2.2.1 ALSO uses SQLite for change history and its last ten
extraction messages. Its scoped DELETE APIs leave those messages behind, so an
API-only reset can let old context influence the next inferred memory write.
The supplied start_mem0 function runs container "mem0" (image "mem0-api"),
publishes 8888:8000, and mounts mem0/server/history at /app/history. This utility
uses the Podman CLI on that same host/connection to run its own embedded helper
inside that container. It reads the active history_db_path from /configure and
opens the existing database there. No host database path or copied helper file
is needed. Missing/incompatible container or database access stops the reset
before any deletion. If you change the endpoint/container constants, they must
still identify the same Mem0 instance.

SQLite's change/audit history, server administration data, the agent's audit log,
backups and the OpenSearch news corpus are preserved. This is a functional
conversation reset, not forensic erasure. Deletion spans two stores and is not
atomic; after a failure keep writers stopped and rerun with the same settings.

All application logic is in this file; there are no project/common imports.
The SDK and server must use the supported 2.2.1 contract. Both installed package
versions are checked. The provided launcher checks out the Git tag but installs
an unpinned package on startup. For a consistent server version, change its
package-install command to:
    pip install -q --force-reinstall --no-deps mem0ai==2.2.1
This utility does not change the server's packages or configuration.
Reference implementation:
https://github.com/mem0ai/mem0/blob/v2.2.1/server/main.py
https://github.com/mem0ai/mem0/blob/v2.2.1/mem0/memory/main.py
https://github.com/mem0ai/mem0/blob/v2.2.1/mem0/memory/storage.py
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
import inspect
import json
import math
import os
import re
import subprocess
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
# This is the container NAME in start_mem0, not its image name (mem0-api).
# Run this utility with access to the same Podman connection that started it.
MEM0_CONTAINER = "mem0"

# The OSS server does not have Mem0 Platform tenancy. These constants satisfy the
# official SDK's eager Project initialization and are used only by this adapter.
ORG_ID = "self-hosted"
PROJECT_ID = "self-hosted"

# Enter the original News Agent identity, before its news-agent: prefix.
# Match the server's MEMORY_USER_ID or authenticated WSGI REMOTE_USER exactly.
MEMORY_USER_ID = "user-456"
MEMORY_SESSION_ID = [
    "456789",
]

BATCH_SIZE = 1000
MEMORY_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
RUN_SUFFIXES = ("complete", "synthesized", "internet", "complete-head")
MEMORY_DOMAIN = "technology_news"


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
    timeout: float = 30.0
    user_id: str = ""
    session_ids: tuple[str, ...] = ()
    container: str = "mem0"

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
    if not isinstance(MEMORY_USER_ID, str) or not MEMORY_USER_ID.strip():
        raise ResetError("MEMORY_USER_ID must be a nonempty string.")
    # Match scoped_memory_settings() exactly, including unconditional prefixing.
    user_id = f"news-agent:{MEMORY_USER_ID.strip()}"

    if not isinstance(MEMORY_SESSION_ID, (list, tuple)):
        raise ResetError("MEMORY_SESSION_ID must be a list or tuple of session IDs.")

    if any(
        not isinstance(session_id, (str, int)) or isinstance(session_id, bool)
        for session_id in MEMORY_SESSION_ID
    ):
        raise ResetError("Session IDs must be strings or integers.")
    session_ids = tuple(str(session_id).strip() for session_id in MEMORY_SESSION_ID)
    if not session_ids or any(not session_id for session_id in session_ids):
        raise ResetError("MEMORY_SESSION_ID must contain at least one nonempty session ID.")
    if len(session_ids) != len(set(session_ids)):
        raise ResetError("MEMORY_SESSION_ID contains duplicate session IDs.")

    try:
        timeout = float(MEM0_TIMEOUT)
    except (TypeError, ValueError) as exc:
        raise ResetError("MEM0_TIMEOUT must be a number greater than zero.") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ResetError("MEM0_TIMEOUT must be finite and greater than zero.")
    if not isinstance(MEM0_CONTAINER, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", MEM0_CONTAINER.strip(),
    ):
        raise ResetError("MEM0_CONTAINER must be the Mem0 API container's name or ID.")

    return ResetSettings(
        api_url=server_url(str(MEM0_API_URL)),
        api_key=str(MEM0_API_KEY).strip(),
        timeout=timeout,
        user_id=user_id,
        session_ids=session_ids,
        container=MEM0_CONTAINER.strip(),
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
        self.server_config: dict[str, Any] = {}
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
            self.server_config = config

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
def connect(settings: ResetSettings) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Use the official remote SDK; never instantiate a local Memory engine."""
    installed = version("mem0ai")
    if installed != SDK_VERSION:
        raise ResetError(f"Install mem0ai=={SDK_VERSION}; found {installed}.")

    os.environ["MEM0_TELEMETRY"] = "false"
    from mem0 import MemoryClient

    transport = ResetTransport(settings)
    with httpx.Client(
        base_url=settings.api_url,
        transport=transport,
        timeout=settings.timeout,
        follow_redirects=False,
        trust_env=False,
    ) as http:
        # MemoryClient requires a nonempty API key even when the OSS server has
        # authentication disabled. The placeholder is consumed only by the SDK;
        # ResetTransport never forwards it to the OSS server.
        client = MemoryClient(
            host=settings.api_url,
            api_key=settings.api_key or "local-auth-disabled",
            client=http,
        )
        yield client, transport.server_config


def message_scopes(settings: ResetSettings) -> tuple[str, ...]:
    """Match Mem0 2.2.1's delimiter escaping and sorted entity-key order."""
    def escape(value: str) -> str:
        return value.replace("%", "%25").replace("&", "%26").replace("=", "%3D")

    return tuple(
        f"run_id={escape(run_id)}&user_id={escape(settings.user_id)}"
        for session_id in settings.session_ids
        for run_id in settings.run_ids_for_session(session_id)
    )


def history_worker() -> None:
    """Run this function's source inside the Mem0 container using only stdlib.

    Inputs arrive as JSON on stdin. No project/server imports, engine creation,
    helper files, package changes, or SQL string interpolation are needed.
    """
    from importlib.metadata import version
    import json
    import math
    from pathlib import Path
    import sqlite3
    import sys

    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("Expected a history-operation JSON object.")
        installed = version("mem0ai")
        expected_version = request["expected_version"]
        if installed != expected_version:
            raise RuntimeError(
                f"Mem0 container has mem0ai=={installed}; this reset requires "
                f"mem0ai=={expected_version}. The Git checkout does not pin the "
                "launcher's pip install. Pin its startup install to "
                f"mem0ai=={expected_version} and restart with that version."
            )
        operation = request["operation"]
        if operation not in {"inspect", "clear"}:
            raise ValueError("Unsupported extraction-history operation.")
        scopes = request["scopes"]
        if (
            not isinstance(scopes, list) or not scopes
            or any(not isinstance(scope, str) or not scope for scope in scopes)
            or len(scopes) != len(set(scopes))
        ):
            raise ValueError("Distinct, nonempty exact message scopes are required.")
        timeout = float(request["timeout"])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("History timeout must be finite and positive.")
        db_path = request["history_db_path"]
        if not isinstance(db_path, str) or not db_path.strip() or db_path == ":memory:":
            raise ValueError("The server must report a persistent history database path.")
        # The worker's /app working directory matches start_mem0. Resolve any
        # server-configured relative path in that same directory, not on the host.
        path = Path(db_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("The server history database is not an existing file.")
        stat = path.stat()
        database = {"path": str(path), "device": stat.st_dev, "inode": stat.st_ino}
        previous = request.get("expected_database")
        if previous is not None and previous != database:
            raise RuntimeError("The history database changed during reset; keep writers stopped and rerun.")
        if operation == "clear" and previous is None:
            raise ValueError("Clearing requires a successful database preflight.")

        # mode=rw prevents creation; as_uri() escapes reserved filename characters.
        connection = sqlite3.connect(
            path.as_uri() + "?mode=rw", uri=True, timeout=timeout, isolation_level=None,
        )
        try:
            expected_tables = {
                "messages": {"id", "session_scope", "role", "content", "name", "created_at"},
                "history": {"id", "memory_id", "old_memory", "new_memory", "event"},
            }
            for table, required in expected_tables.items():
                entry = connection.execute(
                    "SELECT type FROM sqlite_master WHERE name = ?", (table,),
                ).fetchone()
                columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                if entry != ("table",) or not required.issubset(columns):
                    raise RuntimeError(f"The database lacks the expected Mem0 {table!r} table.")

            def count_messages() -> int:
                return sum(
                    connection.execute(
                        "SELECT COUNT(*) FROM messages WHERE session_scope = ?", (scope,),
                    ).fetchone()[0]
                    for scope in scopes
                )

            # Inspect checks write access but releases its lock before any REST
            # deletion, which itself must append to this database's audit table.
            connection.execute("BEGIN IMMEDIATE")
            matched_before = count_messages()
            deleted = 0
            if operation == "clear":
                deleted = sum(
                    connection.execute(
                        "DELETE FROM messages WHERE session_scope = ?", (scope,),
                    ).rowcount
                    for scope in scopes
                )
            remaining = count_messages()
            if operation == "clear" and remaining:
                raise RuntimeError("Extraction messages remain after deletion.")
            connection.execute("COMMIT" if operation == "clear" else "ROLLBACK")
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()
        print(json.dumps({
            "ok": True, "version": installed, "database": database,
            "matched_before": matched_before, "deleted": deleted, "remaining": remaining,
        }))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        raise SystemExit(1)


def history_operation(
    settings: ResetSettings,
    server_config: dict[str, Any],
    operation: str,
    expected_database: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect or clear scoped extraction context in the named Podman container."""
    if operation not in {"inspect", "clear"}:
        raise ResetError("Unsupported extraction-history operation.")
    db_path = server_config.get("history_db_path")
    if not isinstance(db_path, str) or not db_path.strip() or db_path == ":memory:":
        raise ResetError("Mem0 /configure must report a persistent history_db_path.")
    request = {
        "operation": operation, "expected_version": SDK_VERSION,
        "history_db_path": db_path, "scopes": message_scopes(settings),
        "timeout": settings.timeout, "expected_database": expected_database,
    }
    # Transmit our own function, not a file in /app that could trigger --reload.
    # All variable values go through JSON stdin; no shell or API key is involved.
    source = inspect.getsource(history_worker) + "\nhistory_worker()\n"
    command = [
        "podman", "exec", "--workdir", "/app", "-i", settings.container,
        "python", "-c", source,
    ]
    try:
        completed = subprocess.run(
            command, input=json.dumps(request), capture_output=True, text=True,
            timeout=settings.timeout, check=False,
        )
    except FileNotFoundError as exc:
        raise ResetError("Podman is required. Run this utility where start_mem0's Podman connection is available.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ResetError(
            "The container history operation timed out. A requested clear may "
            "still finish inside the container; keep writers stopped and rerun."
        ) from exc
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        # Raw container diagnostics can contain sensitive configuration. Report
        # the failed operation without copying arbitrary stdout or stderr.
        raise ResetError(
            f"Mem0 history {operation!r} in container {settings.container!r} "
            f"returned no valid helper response (exit code {completed.returncode}). "
            "Check that the Mem0 container is running and accessible through "
            "this Podman connection."
        ) from exc
    if completed.returncode != 0 or not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error", "Container helper failed.") if isinstance(result, dict) else "Invalid helper response."
        detail = str(detail)[:1000]
        if settings.api_key:
            detail = detail.replace(settings.api_key, "[redacted]")
        raise ResetError(f"Mem0 container history operation failed: {detail}")
    database = result.get("database")
    if (
        result.get("version") != SDK_VERSION
        or not isinstance(database, dict)
        or not isinstance(database.get("path"), str)
        or not database["path"]
        or any(type(database.get(key)) is not int for key in ("device", "inode"))
        or any(type(result.get(key)) is not int or result[key] < 0
               for key in ("matched_before", "deleted", "remaining"))
        or (expected_database is not None and database != expected_database)
        or (operation == "inspect" and result["deleted"] != 0)
        or (operation == "clear" and result["remaining"] != 0)
    ):
        raise ResetError("Mem0 returned an invalid container history result.")
    return result


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
    if metadata.get("memory_domain") != MEMORY_DOMAIN:
        raise ResetError("Mem0 returned a record outside the News Agent memory domain.")

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
        print(f"Session {session_id!r}: {count} memory records deleted.", flush=True)
    return total


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight both stores, show the exact scope, then require confirmation."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args(argv)

    try:
        settings = build_settings()

        print(f"Mem0 endpoint: {settings.api_url}")
        print(f"Mem0 Podman container: {settings.container!r}")
        print(f"Configured identity: {MEMORY_USER_ID.strip()!r}")
        print(f"Effective Mem0 user: {settings.user_id!r}")
        print(f"Memory domain: {MEMORY_DOMAIN!r}")
        print("Sessions and runs to delete:")
        for session_id in settings.session_ids:
            print(f"  Session {session_id!r}")
            for run_id in settings.run_ids_for_session(session_id):
                print(f"    {run_id!r}")

        with connect(settings) as (client, server_config):
            provider = server_config["vector_store"].get("provider", "unknown")
            print(f"Server vector backend: {provider!r}")
            print(f"Server history database: {server_config.get('history_db_path', 'not reported')!r}")
            history = history_operation(settings, server_config, "inspect")
            print(f"Verified SDK/server package: mem0ai=={history['version']}")
            print(f"Verified database inside container: {history['database']['path']!r}")
            print(f"Selected extraction-context messages: {history['remaining']}")
            print("Stop the News Agent server(s) and all other writers to these sessions.")
            print("Let in-flight writes finish; keep the Mem0 server running.")
            print("This clears conversation memories and recent extraction context.")
            print("Other sessions and change/audit history are preserved.")
            if input("Type DELETE to reset all listed sessions, or press Enter to cancel: ").strip() != "DELETE":
                print("Cancelled. No memory was deleted.")
                return 0

            total = reset_sessions(client, settings)
            cleared = history_operation(settings, server_config, "clear", history["database"])
            context_deleted = cleared["deleted"]
            print(f"Deleted {context_deleted} extraction-context messages.", flush=True)
            # Check all sessions again after BOTH stores have been processed.
            for session_id in settings.session_ids:
                for run_id in settings.run_ids_for_session(session_id):
                    if list_batch(client, settings, session_id, run_id):
                        raise ResetError("Memories reappeared during final verification.")
            final_history = history_operation(settings, server_config, "inspect", history["database"])
            if final_history["remaining"]:
                raise ResetError("Extraction messages reappeared during final verification.")

        print(
            f"\nReset complete: deleted {total} memory records and "
            f"{context_deleted} extraction-context messages across "
            f"{len(settings.session_ids)} session(s) for user {settings.user_id!r}."
        )
        print("All selected memory runs and extraction-message scopes verified empty.")
        print("Unlisted sessions remain available to cross-session topic recall.")
        return 0
    except (EOFError, KeyboardInterrupt):
        print(
            "\nReset interrupted. Any completed deletions remain; rerun to finish.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"Memory reset failed: {exc}", file=sys.stderr)
        print("If deletion started, keep writers stopped and rerun to finish.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
