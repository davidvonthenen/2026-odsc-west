"""Run the governed News Agent through Flask's text-only Chat Completions API."""
from __future__ import annotations

import atexit
import logging
import signal
import threading
import time
import uuid

import click
from flask import Flask, Response, g, jsonify, request
from werkzeug.exceptions import BadRequest, HTTPException

from ..common.config import Settings, load_settings
from ..common.logging import MEMORY_LIFECYCLE, MemoryLifecycleLog
from .news_agent import NewsAgent
from .rest_contract import RequestError, authenticated_user, completion_response, parse_request
from .runtime import AgentRuntime, RuntimeClosedError

LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_LEVEL = "info"


def _openai_error(message: str, *, status: int, param: str | None = None,
                  code: str | None = None, error_type: str = "invalid_request_error") -> tuple[Response, int]:
    return jsonify({"error": {"message": message, "type": error_type,
                             "param": param, "code": code}}), status


def create_app(settings: Settings | None = None, agent: NewsAgent | None = None) -> Flask:
    settings = settings or load_settings()
    runtime = AgentRuntime(lambda: agent if agent is not None else NewsAgent(settings=settings))
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    app.extensions["news_agent_runtime"] = runtime
    atexit.register(runtime.close)

    @app.before_request
    def identify_request() -> None:
        # Do not use caller-supplied identifiers as audit authority.
        g.completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        if request.path == "/v1/chat/completions" and request.method == "POST":
            g.memory_lifecycle = MemoryLifecycleLog(g.completion_id)
            g.memory_lifecycle_token = MEMORY_LIFECYCLE.set(g.memory_lifecycle)

    @app.after_request
    def response_headers(response: Response) -> Response:
        response.headers["x-request-id"] = g.completion_id
        response.headers["Cache-Control"] = "no-store"
        if hasattr(g, "memory_lifecycle"):
            g.memory_lifecycle.summary(
                http_status=response.status_code,
                outcome=getattr(g, "memory_outcome", "request_error"),
                **getattr(g, "memory_summary_details", {}),
            )
        return response

    @app.teardown_request
    def clear_memory_lifecycle(exc: BaseException | None) -> None:
        token = g.pop("memory_lifecycle_token", None)
        if token is not None:
            MEMORY_LIFECYCLE.reset(token)

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        response, status = _openai_error(
            exc.description, status=exc.code or 500,
            code=f"http_{exc.code or 500}",
        )
        # Preserve required protocol headers, such as Allow for HTTP 405.
        for key, value in exc.get_headers():
            if key.lower() not in {"content-type", "content-length"}:
                response.headers[key] = value
        return response, status

    @app.get("/health")
    def health():
        # Liveness only: this does not connect to or probe dependent services.
        return jsonify({
            "status": "ok", "agent": "News Agent", "interface": "openai-chat-completions",
            "model": settings.public_model_name, "streaming": False,
            "historical_index": settings.opensearch_index,
            "tavily_mcp_url": settings.tavily_mcp_url,
            "router_verifier_model": settings.orch_model, "generator_model": settings.llm_model,
            "memory_required": True, "mem0_api_url": settings.mem0_api_url,
            "ner_required": True, "ner_server_url": settings.ner_url,
            "compact_k": settings.compact_k,
            "memory_expiration": settings.memory_expiration,
        })

    @app.get("/v1/models")
    def models():
        return jsonify({"object": "list", "data": [{
            "id": settings.public_model_name, "object": "model", "created": 0,
            "owned_by": "news-agent",
        }]})

    @app.post("/v1/chat/completions")
    def chat_completions():
        if not request.is_json:
            return _openai_error("Content-Type must be application/json.", status=415,
                                 param="request_body", code="unsupported_media_type")
        try:
            payload = request.get_json()
        except BadRequest:
            return _openai_error("The body is not valid JSON.", status=400,
                                 param="request_body", code="invalid_json")
        try:
            parsed, question, session_id = parse_request(payload, settings)
            user_id = authenticated_user(request.environ)
        except RequestError as exc:
            return _openai_error(str(exc), status=exc.status, param=exc.param, code=exc.code)
        metadata = {"client_id": "openai-rest", "request_id": g.completion_id}
        if user_id is not None:
            metadata["user_id"] = user_id
        try:
            result = runtime.invoke(question, session_id, user_id=user_id,
                                    request_metadata=metadata, messages=parsed.text_messages(),
                                    compact_k=parsed.compact_k,
                                    memory_expiration=parsed.memory_expiration)
            g.memory_outcome = result.get("status", "unknown")
            g.memory_summary_details = {
                "session_id": session_id, "audit_id": result.get("audit_id"),
                "memory_status": result.get("memory_status", "not_run"),
                "history_status": result.get("history_status", "not_run"),
                "version": result.get("version"), "compacted": result.get("compacted", False),
            }
            response = completion_response(
                result, completion_id=g.completion_id, model=settings.public_model_name,
                session_id=session_id, created=int(time.time()),
            )
        except RuntimeClosedError:
            g.memory_outcome = "agent_unavailable"
            return _openai_error("The News Agent is shutting down.", status=503,
                                 error_type="server_error", code="agent_unavailable")
        except Exception:
            g.memory_outcome = "agent_failure"
            LOGGER.exception("news_rest_request_failed request_id=%s", g.completion_id)
            return _openai_error("The News Agent could not complete the governed request.",
                                 status=500, error_type="server_error", code="agent_failure")
        return jsonify(response)

    return app


def main(host: str | None = None, port: int | None = None,
         log_level: str = DEFAULT_LOG_LEVEL) -> None:
    settings = load_settings()
    if host is not None:
        settings.api_host = host
    if port is not None:
        settings.api_port = port
    logging.basicConfig(level=log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = create_app(settings)
    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        # Convert normal termination to stack unwinding so accepted turns drain.
        def terminate(signum, frame):
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, terminate)
    try:
        app.run(host=settings.api_host, port=settings.api_port,
                threaded=True, debug=False, use_reloader=False)
    finally:
        app.extensions["news_agent_runtime"].close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


@click.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=click.IntRange(1, 65535))
@click.option("--log-level", default=DEFAULT_LOG_LEVEL,
              type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False))
def cli(host: str | None, port: int | None, log_level: str) -> None:
    main(host, port, log_level)


if __name__ == "__main__":
    cli()
