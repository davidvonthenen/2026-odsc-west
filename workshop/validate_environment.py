#!/usr/bin/env python3
"""Check the workshop stack without creating, changing, or deleting memories.

Place beside setup.sh and run after setup.sh has started the containers.
Exported shell variables take precedence over optional workshop.env, then .env,
then the host-side defaults from setup.sh. Env files are read beside this script;
the container-only mem0/server/.env is deliberately not loaded.
OPENAI_BASE_URL defaults to https://api.openai.com/v1 and can point to an
OpenAI-compatible API. OPENAI_MODEL and MEM0_DEFAULT_EMBEDDER_MODEL select the
chat and embedding models; MEM0_DEFAULT_LLM_MODEL checks Mem0's configured model.
OpenAI-compatible API checks make small billable requests; Tavily uses GET /usage.
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
from urllib.parse import urlsplit

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


class ConfigurationError(CheckFailure):
    """A named environment setting is invalid, rather than a programming error."""


def env_value(name: str, default: str) -> str:
    """Use setup defaults for missing or blank non-secret settings."""
    return os.getenv(name, "").strip() or default


def endpoint(name: str, value: str) -> str:
    """Validate an HTTP(S) base URL without including its value in errors."""
    try:
        parsed = urlsplit(value)
        invalid = (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or "?" in value or "#" in value or "\\" in value
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
            or parsed.netloc.endswith(":")
        )
    except ValueError:
        invalid = True
    if invalid:
        raise ConfigurationError(
            f"{name} must be an HTTP(S) URL with a valid hostname and port, "
            "without embedded credentials, whitespace, query parameters, or fragments."
        ) from None
    return value.rstrip("/")


def host_value(name: str, default: str) -> str:
    """Accept a bare hostname or IP address, not a URL or host:port pair."""
    value = env_value(name, default)
    try:
        authority = f"[{value}]" if ":" in value and not value.startswith("[") else value
        parsed = urlsplit("//" + authority)
        invalid = (
            not parsed.hostname or parsed.port is not None
            or parsed.username is not None or parsed.password is not None
            or any(char in value for char in "/\\?#")
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
            or authority.endswith(":")
        )
    except ValueError:
        invalid = True
    if invalid:
        raise ConfigurationError(
            f"{name} must be a hostname or IP address without a scheme, port, or path."
        ) from None
    return parsed.hostname


@dataclass(frozen=True)
class Result:
    status: str
    detail: str


@dataclass(frozen=True)
class Settings:
    opensearch_url: str
    mem0_url: str
    mem0_api_key: str
    mem0_openai_model: str
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    app_db: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_embedding_model: str
    tavily_api_key: str

    @classmethod
    def from_env(cls) -> "Settings":
        def port(name: str, default: int) -> int:
            try:
                value = int(env_value(name, str(default)))
            except ValueError:
                raise ConfigurationError(
                    f"{name} must be an integer between 1 and 65535."
                ) from None
            if not 1 <= value <= 65535:
                raise ConfigurationError(f"{name} must be between 1 and 65535.")
            return value

        search_host = host_value("OPENSEARCH_HOST", "127.0.0.1")
        search_authority = f"[{search_host}]" if ":" in search_host else search_host
        return cls(
            opensearch_url=endpoint(
                "OPENSEARCH_HOST / OPENSEARCH_PORT",
                f"http://{search_authority}:{port('OPENSEARCH_PORT', 9200)}",
            ),
            mem0_url=endpoint(
                "MEM0_BASE_URL", env_value("MEM0_BASE_URL", "http://127.0.0.1:8888")
            ),
            mem0_api_key=os.getenv("MEM0_API_KEY", "").strip(),
            mem0_openai_model=env_value("MEM0_DEFAULT_LLM_MODEL", "gpt-5.4-mini"),
            postgres_host=host_value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", 8432),
            postgres_user=env_value("POSTGRES_USER", "postgres"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "postgrespostgres"),
            postgres_db=env_value("POSTGRES_DB", "postgres"),
            app_db=env_value("APP_DB_NAME", "mem0_app"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_base_url=endpoint("OPENAI_BASE_URL", env_value("OPENAI_BASE_URL", OPENAI_URL)),
            openai_model=env_value("OPENAI_MODEL", "gpt-5.4-mini"),
            openai_embedding_model=env_value(
                "MEM0_DEFAULT_EMBEDDER_MODEL", "text-embedding-3-small"
            ),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        )


def require_key(name: str, value: str) -> None:
    lowered = value.strip().lower()
    if not lowered or lowered in {"replace-me", "not-needed", "changeme"} or "your-key" in lowered:
        raise CheckFailure(
            f"{name} is missing or contains a placeholder. Export it in the shell "
            "that launches Python, or set it in workshop.env or .env beside this script. "
            "A .bash_profile assignment must be exported and loaded into that shell."
        )


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
        advice = {
            401: "Authentication rejected; verify the API key for this service.",
            403: "Access denied; verify API permissions for this service.",
            404: "Endpoint not found; verify the service base URL and API path.",
            429: "Rate or quota limit reached; check API usage and limits.",
        }
        raise CheckFailure(
            f"HTTP {response.status_code}; "
            + advice.get(response.status_code, "Check the endpoint and service logs.")
        )
    try:
        data = response.json()
    except ValueError:
        raise CheckFailure("Expected a JSON response, not an HTML page or proxy response.") from None
    if not isinstance(data, dict):
        raise CheckFailure("Expected a JSON object; verify the service base URL and API path.")
    return data


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
    return OpenAI(
        api_key=s.openai_api_key, base_url=s.openai_base_url, timeout=30.0, max_retries=0
    )


def unsupported_completion_limit(exc: Exception) -> bool:
    """Allow a bounded compatibility retry only for an unsupported token field."""
    if getattr(exc, "status_code", None) not in {400, 422}:
        return False
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    error = body.get("error", body)
    if not isinstance(error, dict):
        return False
    message = str(error.get("message", "")).lower()
    field_named = (
        error.get("param") == "max_completion_tokens"
        or "max_completion_tokens" in message
    )
    unsupported = error.get("code") in {"unsupported_parameter", "unknown_parameter"} or any(
        term in message for term in (
            "not supported", "unsupported", "unknown parameter", "unrecognized",
            "unexpected keyword", "not permitted", "not allowed",
        )
    )
    return field_named and unsupported


def check_openai_chat(s: Settings) -> Result:
    from openai import APIStatusError

    with cloud_client(s) as client:
        request = {
            "model": s.openai_model,
            "messages": [{"role": "user", "content": "Reply OK."}],
        }
        try:
            response = client.chat.completions.create(**request, max_completion_tokens=16)
        except APIStatusError as exc:
            if not unsupported_completion_limit(exc):
                raise
            response = client.chat.completions.create(**request, max_tokens=16)
    if not response.id or not response.choices:
        raise CheckFailure("The API returned no completion identifier or choices.")
    return Result(
        "PASS", f"Configured OpenAI-compatible API accepted a chat request for {s.openai_model}. "
        "A short token budget may produce empty text; this checks access, not answer quality."
    )


def check_openai_embeddings(s: Settings) -> Result:
    with cloud_client(s) as client:
        response = client.embeddings.create(
            model=s.openai_embedding_model, input="Workshop connectivity check."
        )
    if not response.data or not response.data[0].embedding:
        raise CheckFailure("The API returned no embedding.")
    return Result("PASS", f"API embedding request succeeded for {s.openai_embedding_model}.")


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


def safe_error(exc: Exception) -> str:
    """Do not expose keys, response bodies, request headers, or arbitrary exception text."""
    if isinstance(exc, CheckFailure):
        return str(exc)
    status = getattr(exc, "status_code", None)
    advice = {
        400: "Request rejected; verify the configured model and its supported API parameters.",
        401: "Authentication rejected; verify the relevant key without printing it.",
        403: "Access denied; verify project permissions and model access.",
        404: "Resource or model not found; verify the configured endpoint and model.",
        422: "Request validation failed; verify the provider's supported API parameters.",
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
        print("Run: python -m pip install -r requirements.txt")
        return 2

    from dotenv import load_dotenv

    try:
        load_dotenv(ROOT / "workshop.env", override=False)
        load_dotenv(ROOT / ".env", override=False)
    except (OSError, UnicodeError):
        print("[FAIL] Cannot read workshop.env or .env beside this script; check access and UTF-8 encoding.")
        return 2
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"[FAIL] Configuration: {exc}")
        return 2
    except Exception as exc:
        print(
            f"[FAIL] Validator initialization failed ({type(exc).__name__}); "
            "inspect the Settings implementation. Service and API-key checks have not run."
        )
        return 2

    print("Workshop validation: no memory writes, resets, or index changes.")
    print("Configuration: exported environment > workshop.env > .env > setup defaults.")
    print("Env files are optional and read beside this script.")
    print("OpenAI-compatible chat and embedding probes make small billable requests.")
    print("Podman services and API providers are checked separately.\n")
    checks = (
        ("Host tools and Podman", check_tools),
        ("Workshop containers", check_containers),
        ("OpenSearch and k-NN", check_opensearch),
        ("PostgreSQL and pgvector", check_pgvector),
        ("Mem0 application database", check_app_database),
        ("Mem0 API and vector-store read", check_mem0),
        ("OpenAI-compatible API key and model access", check_openai_chat),
        ("OpenAI-compatible embedding access", check_openai_embeddings),
        ("Tavily API key", check_tavily),
    )
    results = [run_check(label, lambda fn=fn: fn(settings)) for label, fn in checks]
    return summarize(results)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nValidation interrupted.")
        raise SystemExit(130)
