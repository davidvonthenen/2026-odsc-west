"""REST client for the News Agent's text-only Chat Completions endpoint."""
from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .common.config import Settings, load_settings


def normalize_api_url(url: str, fallback: str) -> str:
    """Accept an origin, /v1 base URL, or full Chat Completions endpoint."""
    value = (url or fallback).strip().rstrip("/")
    parts = urlsplit(value)
    try:
        _ = parts.port
    except ValueError as exc:
        raise ValueError("API URL has an invalid port") from exc
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):
        raise ValueError("API URL must be an HTTP(S) URL without credentials, query, or fragment")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[:-len("/chat/completions")]
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def chat_completion(client: httpx.Client, *, model: str,
                    messages: list[dict[str, str]], session_id: str) -> dict[str, Any]:
    """Send one non-streaming request, without automatic POST retries."""
    if not session_id.strip():
        raise ValueError("session_id must not be empty")
    response = client.post("chat/completions", json={
        "model": model, "messages": messages, "stream": False, "user": session_id,
    })
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"News Agent returned non-JSON HTTP {response.status_code}") from exc
    if response.is_error:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        detail = error.get("message") if isinstance(error, dict) else None
        raise RuntimeError(f"News Agent HTTP {response.status_code}: {detail or response.reason_phrase}")
    if not isinstance(payload, dict) or payload.get("object") != "chat.completion":
        raise RuntimeError("News Agent returned an invalid chat completion")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("News Agent returned no assistant content") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("News Agent returned empty assistant content")
    return payload


def create_client(settings: Settings, *, url: str | None = None,
                  timeout: float = 1200.0) -> httpx.Client:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and greater than 0")
    base_url = normalize_api_url(url or settings.api_url, settings.api_url)
    return httpx.Client(base_url=base_url + "/", timeout=httpx.Timeout(timeout),
                        follow_redirects=False)


def ask_news_agent(question: str, *, url: str | None = None, timeout: float = 1200.0,
                   session_id: str | None = None) -> str:
    """Query the running HTTP server, never instantiate the agent in the client."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    settings = load_settings()
    with create_client(settings, url=url, timeout=timeout) as client:
        payload = chat_completion(
            client, model=settings.public_model_name,
            messages=[{"role": "user", "content": question}],
            session_id=session_id if session_id is not None else settings.mem0_session_id,
        )
    return payload["choices"][0]["message"]["content"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Query the News Agent over its REST API")
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--url", default=settings.api_url,
                        help="Agent origin, /v1 base URL, or Chat Completions URL")
    parser.add_argument("--timeout", type=float, default=1200.0,
                        help="HTTP timeout in seconds (default: 1200)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        print(ask_news_agent(args.question, url=args.url, timeout=args.timeout))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        raise SystemExit(f"News Agent query failed: {exc}") from exc


if __name__ == "__main__":
    main()
