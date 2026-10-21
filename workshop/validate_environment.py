#!/usr/bin/env python3
"""Check the workshop stack without creating, changing, or deleting memories.

Run from the workshop project root after setup.sh has started the containers.
Shell variables take precedence over workshop.env, then .env, then defaults.
OpenAI checks make small billable requests; Tavily uses GET /usage.
Mem0 may record ordinary API access logs for the read-only requests.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parent
HTTP_TIMEOUT = 10.0
COMMAND_TIMEOUT = 20
OPENAI_URL = "https://api.openai.com/v1"
CONTAINERS = (
    "postgres",
    "mem0",
    "mem0-dashboard",
    "opensearch-single",
    "opensearch-single-dashboards",
)


class CheckFailure(RuntimeError):
    """An actionable diagnostic containing no credentials or response bodies."""


@dataclass(frozen=True)
class Result:
    status: str
    detail: str


@dataclass(frozen=True)
class Settings:
    opensearch_url: str
    opensearch_index: str
    mem0_url: str
    mem0_api_key: str
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    app_db: str
    openai_api_key: str
    openai_model: str
    openai_embedding_model: str
    tavily_api_key: str
    # llm_url: str
    # llm_api_key: str
    # llm_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        def port(name: str, default: int) -> int:
            value = int(os.getenv(name, str(default)))
            if not 1 <= value <= 65535:
                raise ValueError(f"{name} must be between 1 and 65535.")
            return value

        settings = cls(
            opensearch_url=(
                f"http://{os.getenv('OPENSEARCH_HOST', '127.0.0.1')}:"
                f"{port('OPENSEARCH_PORT', 9200)}"
            ),
            opensearch_index=os.getenv("OPENSEARCH_INDEX", "bbc-vector-chunks"),
            mem0_url=os.getenv("MEM0_BASE_URL", "http://127.0.0.1:8888").rstrip("/"),
            mem0_api_key=os.getenv("MEM0_API_KEY", ""),
            mem0_openai_model=os.getenv("MEM0_DEFAULT_LLM_MODEL", "gpt-5.4-mini"),
            postgres_host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", 8432),
            postgres_user=os.getenv("POSTGRES_USER", "postgres"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "postgrespostgres"),
            postgres_db=os.getenv("POSTGRES_DB", "postgres"),
            app_db=os.getenv("APP_DB_NAME", "mem0_app"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            openai_embedding_model=os.getenv(
                "MEM0_DEFAULT_EMBEDDER_MODEL", "text-embedding-3-small"
            ),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            # llm_url=os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
            # llm_api_key=os.getenv("LLM_SERVER_API_KEY", "not-needed") or "not-needed",
            # llm_model=os.getenv("LLM_SERVER_MODEL", "local-llm"),
        )
        # for value in (settings.opensearch_url, settings.mem0_url, settings.llm_url):
        #     parsed = urlsplit(value)
        #     if (
        #         parsed.scheme not in {"http", "https"} or not parsed.hostname
        #         or parsed.username or parsed.password or parsed.query or parsed.fragment
        #     ):
        #         raise ValueError("Use HTTP(S) endpoint URLs without embedded credentials.")
        # if not settings.llm_url.endswith("/v1"):
        #     raise ValueError("LLM_SERVER_URL must end in /v1 for this workshop.")
        # if urlsplit(settings.llm_url).hostname == "api.openai.com":
        #     raise ValueError(
        #         "LLM_SERVER_URL must target the workshop's local inference service, "
        #         "not the hosted OpenAI API."
        #     )
        return settings


def require_key(name: str, value: str) -> None:
    lowered = value.strip().lower()
    if not lowered or lowered in {"replace-me", "not-needed", "changeme"} or "your-key" in lowered:
        raise CheckFailure(f"Set {name} in the shell or workshop.env; do not print the key.")


def command(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=COMMAND_TIMEOUT, check=False
        )
    except FileNotFoundError:
        raise CheckFailure(f"Install {args[0]} and ensure it is on PATH.") from None
    except subprocess.TimeoutExpired:
        raise CheckFailure("Command timed out; inspect Podman and its machine.") from None
    if result.returncode:
        # Never echo command output: other command forms could contain credentials.
        raise CheckFailure(
            f"{' '.join(args[:2])} failed (exit {result.returncode}). "
            "Run the documented diagnostic command locally."
        )
    return result.stdout.strip()


def get_json(url: str, *, headers: dict | None = None, params: dict | None = None):
    import httpx

    response = httpx.get(
        url, headers=headers, params=params, timeout=HTTP_TIMEOUT, follow_redirects=False
    )
    if response.status_code != 200:
        raise CheckFailure(
            f"HTTP {response.status_code}; check the endpoint, authentication, and service logs."
        )
    try:
        return response.json()
    except ValueError:
        raise CheckFailure("Expected a JSON response, not an HTML page or proxy response.") from None


def check_tools(_: Settings) -> Result:
    missing = [name for name in ("bash", "git", "curl", "openssl", "make", "python3")
               if shutil.which(name) is None]
    if missing:
        raise CheckFailure("Missing host tools: " + ", ".join(missing))
    command(["podman", "info", "--format", "{{.Host.OS}}"])
    return Result("PASS", "Host tools found; Podman engine responds.")


def check_containers(_: Settings) -> Result:
    output = command([
        "podman", "container", "inspect", "--format", "{{.Name}} {{.State.Status}}",
        *CONTAINERS,
    ])
    states = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2:
            states[fields[0].lstrip("/")] = fields[1]
    bad = [name for name in CONTAINERS if states.get(name) != "running"]
    if bad:
        raise CheckFailure("Missing or stopped containers: " + ", ".join(bad))
    return Result("PASS", "All five containers from setup.sh are running.")


def check_opensearch(s: Settings) -> Result:
    info = get_json(s.opensearch_url + "/")
    if info.get("version", {}).get("distribution") != "opensearch":
        raise CheckFailure("The endpoint did not identify itself as OpenSearch.")
    health = get_json(
        s.opensearch_url + "/_cluster/health",
        params={"wait_for_status": "yellow", "timeout": "8s"},
    )
    status = health.get("status")
    if health.get("timed_out") or status not in {"green", "yellow"}:
        raise CheckFailure("OpenSearch is not ready: cluster health must be green or yellow.")
    nodes = get_json(s.opensearch_url + "/_nodes/plugins").get("nodes", {})
    if not nodes or any(
        not any(p.get("name") == "opensearch-knn" for p in node.get("plugins", []))
        for node in nodes.values()
    ):
        raise CheckFailure("OpenSearch k-NN plugin is missing from one or more nodes.")
    version = info.get("version", {}).get("number", "unknown")
    note = "; yellow is acceptable for a single-node lab" if status == "yellow" else ""
    if version != "3.5.0":
        return Result("WARN", f"OpenSearch {version}, health={status}; setup.sh specifies 3.5.0.")
    return Result("PASS", f"OpenSearch {version}; k-NN available; health={status}{note}.")


def connect_postgres(s: Settings, database: str):
    import psycopg

    # A published host port is required here; 'postgres:5432' is container-only.
    return psycopg.connect(
        host=s.postgres_host, port=s.postgres_port, user=s.postgres_user,
        password=s.postgres_password, dbname=database, connect_timeout=8,
        autocommit=True,
        options="-c default_transaction_read_only=on -c statement_timeout=8000",
    )


def check_pgvector(s: Settings) -> Result:
    with connect_postgres(s, s.postgres_db) as connection:
        with connection.cursor() as cur:
            cur.execute("SELECT current_database(), current_setting('server_version')")
            database, version = cur.fetchone()
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            extension = cur.fetchone()
            if extension is None:
                raise CheckFailure(
                    "The vector extension is not enabled in the memory database. "
                    "Check Mem0 initialization; the validator does not create extensions."
                )
            cur.execute("SELECT '[1,2,3]'::vector <-> '[1,2,3]'::vector")
            if cur.fetchone()[0] != 0:
                raise CheckFailure("The pgvector distance calculation returned an unexpected result.")
    return Result(
        "PASS", f"PostgreSQL {version}; database={database}; pgvector {extension[0]}; SQL works."
    )


def check_app_database(s: Settings) -> Result:
    with connect_postgres(s, s.app_db) as connection:
        with connection.cursor() as cur:
            cur.execute("SELECT version_num FROM alembic_version")
            revision = cur.fetchone()
            if not revision:
                raise CheckFailure("Mem0 application database has no Alembic revision.")
    return Result("PASS", "mem0_app connection and migration marker verified; no user records read.")


def check_mem0(s: Settings) -> Result:
    headers = {"X-API-Key": s.mem0_api_key} if s.mem0_api_key else {}
    status = get_json(s.mem0_url + "/auth/setup-status")
    if status.get("needsSetup") is not False:
        raise CheckFailure("Mem0 admin bootstrap is incomplete; inspect setup.sh seed output.")
    config = get_json(s.mem0_url + "/configure", headers=headers)
    vector = config.get("vector_store", {})
    if vector.get("provider") != "pgvector":
        raise CheckFailure("Mem0's active vector provider is not pgvector; inspect configuration drift.")
    if vector.get("config", {}).get("dbname") != s.postgres_db:
        raise CheckFailure("Mem0's configured database differs from POSTGRES_DB used by the validator.")
    for section, model in (("llm", s.mem0_openai_model), ("embedder", s.openai_embedding_model)):
        item = config.get(section, {})
        if item.get("provider") != "openai" or item.get("config", {}).get("model") != model:
            raise CheckFailure(
                f"Mem0 {section} differs from the expected OpenAI configuration. "
                "Align the validator environment with the running server before proceeding."
            )
    # Use a unique scope; never list all attendees' or all existing memories.
    response = get_json(
        s.mem0_url + "/memories", headers=headers,
        params={"user_id": "workshop-check-" + uuid.uuid4().hex, "top_k": 1},
    )
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise CheckFailure("Mem0 returned an unexpected memory-list response.")
    return Result(
        "PASS", "Bootstrap, active pgvector configuration, and scoped memory read verified; "
        "no memory was created."
    )


def cloud_client(s: Settings):
    from openai import OpenAI

    require_key("OPENAI_API_KEY", s.openai_api_key)
    # Explicit URL prevents OPENAI_BASE_URL from silently validating a local server instead.
    return OpenAI(
        api_key=s.openai_api_key, base_url=OPENAI_URL, timeout=30.0, max_retries=0
    )


def check_openai_chat(s: Settings) -> Result:
    with cloud_client(s) as client:
        response = client.chat.completions.create(
            model=s.openai_model,
            messages=[{"role": "user", "content": "Reply OK."}],
            max_completion_tokens=16,
        )
    if not response.id or not response.choices:
        raise CheckFailure("OpenAI returned no completion identifier or choices.")
    return Result(
        "PASS", f"Hosted OpenAI accepted a chat request for {s.openai_model}. "
        "A short token budget may produce empty text; this checks access, not answer quality."
    )


def check_openai_embeddings(s: Settings) -> Result:
    with cloud_client(s) as client:
        response = client.embeddings.create(
            model=s.openai_embedding_model, input="Workshop connectivity check."
        )
    if not response.data or not response.data[0].embedding:
        raise CheckFailure("OpenAI returned no embedding.")
    return Result("PASS", f"Hosted embedding request succeeded for {s.openai_embedding_model}.")


def check_tavily(s: Settings) -> Result:
    require_key("TAVILY_API_KEY", s.tavily_api_key)
    data = get_json(
        "https://api.tavily.com/usage",
        headers={"Authorization": "Bearer " + s.tavily_api_key},
    )
    if not isinstance(data.get("key"), dict) or not isinstance(data.get("account"), dict):
        raise CheckFailure("Tavily returned an unexpected usage response.")
    key = data["key"]
    limit, usage = key.get("limit"), key.get("usage")
    if isinstance(limit, (int, float)) and isinstance(usage, (int, float)):
        if limit > 0 and usage >= limit:
            raise CheckFailure("Tavily authentication succeeded, but the key's usage limit is reached.")
    return Result("PASS", "Tavily authenticated GET /usage; no web search or MCP request was made.")


# def check_local_llm(s: Settings) -> Result:
#     from openai import OpenAI

#     health = get_json(s.llm_url.removesuffix("/v1") + "/health")
#     if health.get("status") != "ok":
#         raise CheckFailure("The local LLM health endpoint did not report status=ok.")
#     if health.get("model") != s.llm_model:
#         raise CheckFailure("LLM_SERVER_MODEL does not match the running local service.")
#     with OpenAI(
#         base_url=s.llm_url, api_key=s.llm_api_key, timeout=120.0, max_retries=0
#     ) as client:
#         response = client.chat.completions.create(
#             model=s.llm_model,
#             messages=[{"role": "user", "content": "Reply with the word ready."}],
#             max_tokens=16, temperature=0,
#         )
#     if not response.choices or not (response.choices[0].message.content or "").strip():
#         raise CheckFailure("Local inference returned no text; inspect model and service logs.")
#     return Result("PASS", "Local service health and SDK chat inference verified independently of OpenAI.")


# def check_index(s: Settings) -> Result:
#     import httpx

#     base = s.opensearch_url + "/" + quote(s.opensearch_index, safe="")
#     response = httpx.get(base + "/_mapping", timeout=HTTP_TIMEOUT, follow_redirects=False)
#     if response.status_code == 404:
#         return Result("WARN", "RAG index does not exist yet. Expected before make ingest.")
#     if response.status_code != 200:
#         raise CheckFailure("Cannot inspect RAG index mapping; inspect OpenSearch.")
#     mapping = response.json()
#     properties = mapping.get(s.opensearch_index, {}).get("mappings", {}).get("properties", {})
#     if properties.get("embedding", {}).get("type") != "knn_vector":
#         raise CheckFailure("Existing RAG index does not have an embedding knn_vector field.")
#     settings = get_json(base + "/_settings")
#     knn = settings.get(s.opensearch_index, {}).get("settings", {}).get("index", {}).get("knn")
#     if str(knn).lower() != "true":
#         raise CheckFailure("Existing RAG index does not have index.knn enabled.")
#     count = get_json(base + "/_count").get("count")
#     if not isinstance(count, int):
#         raise CheckFailure("OpenSearch returned an invalid document count.")
#     if count == 0:
#         return Result("WARN", "RAG index is empty. Run make ingest before Exercise 1 queries.")
#     return Result("PASS", f"RAG index contains {count} chunks; vector mapping enabled.")


def safe_error(exc: Exception) -> str:
    """Do not expose keys, response bodies, request headers, or arbitrary exception text."""
    if isinstance(exc, CheckFailure):
        return str(exc)
    status = getattr(exc, "status_code", None)
    advice = {
        401: "Authentication rejected; verify the relevant key without printing it.",
        403: "Access denied; verify project permissions and model access.",
        404: "Resource or model not found; verify the configured endpoint and model.",
        429: "Rate or quota limit reached; check API usage, billing, and rate limits.",
    }
    if status in advice:
        return f"HTTP {status}: {advice[status]}"
    name = type(exc).__name__
    if "Timeout" in name or "Connection" in name or "Connect" in name:
        return "Connection failed or timed out; verify service startup, address, proxy, and firewall."
    if name in {"OperationalError", "InvalidPassword", "InvalidCatalogName", "UndefinedTable"}:
        return f"{name}: verify PostgreSQL credentials, database initialization, and migrations."
    return f"{name}: check the relevant local logs. Raw exception details are suppressed."


def run_check(label: str, operation: Callable[[], Result]) -> Result:
    print(f"[CHECK] {label}", flush=True)
    try:
        result = operation()
    except Exception as exc:
        result = Result("FAIL", safe_error(exc))
    print(f"[{result.status}] {label}: {result.detail}", flush=True)
    return result


def summarize(results: list[Result]) -> int:
    counts = {status: sum(r.status == status for r in results) for status in ("PASS", "WARN", "FAIL")}
    print(f"\nSummary: {counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed.")
    if counts["FAIL"]:
        print("Environment is not ready. Resolve failures and rerun this script.")
        return 1
    print("Service checks passed. Resolve warnings before the steps they affect.")
    print("This is not a memory-write test, a corpus audit, or an end-to-end RAG evaluation.")
    return 0


def main() -> int:
    if sys.version_info < (3, 10):
        print("[FAIL] Python 3.10 or newer is required.")
        return 2
    missing = []
    for name in ("dotenv", "httpx", "openai", "psycopg"):
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        print("[FAIL] Missing validation dependencies: " + ", ".join(missing))
        print("Run: python -m pip install -r requirements-validation.txt")
        return 2

    from dotenv import load_dotenv

    load_dotenv(ROOT / "workshop.env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    try:
        settings = Settings.from_env()
    except (ValueError, TypeError):
        print("[FAIL] Invalid endpoint or port configuration; review workshop.env and .env.")
        return 2

    print("Workshop validation: no memory writes, resets, or index changes.")
    print("OpenAI chat and embedding probes make small billable requests.")
    print("Local services and cloud providers are checked separately.\n")
    checks = (
        ("Host tools and Podman", check_tools),
        ("Workshop containers", check_containers),
        ("OpenSearch and k-NN", check_opensearch),
        ("PostgreSQL and pgvector", check_pgvector),
        ("Mem0 application database", check_app_database),
        ("Mem0 API and vector-store read", check_mem0),
        ("OpenAI API key and model access", check_openai_chat),
        ("OpenAI embedding access", check_openai_embeddings),
        ("Tavily API key", check_tavily),
        # ("Local inference service", check_local_llm),
        # ("RAG index readiness", check_index),
    )
    results = [run_check(label, lambda fn=fn: fn(settings)) for label, fn in checks]
    return summarize(results)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nValidation interrupted.")
        raise SystemExit(130)
