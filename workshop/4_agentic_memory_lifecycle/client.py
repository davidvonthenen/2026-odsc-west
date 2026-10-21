"""Interactive OpenAI SDK client with server-returned compacted chat history."""
from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from typing import Any

from src.common.config import load_settings
from src.query import normalize_api_url


def send_turn(
    client: Any,
    *,
    model: str,
    history: list[dict[str, str]],
    question: str,
    session_id: str,
    compact_k: int = 5,
    memory_expiration: float = 5.0,
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    """Return a new history only after a valid response; never mutate the input."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    pending = [dict(message) for message in history]
    pending.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=model, messages=pending, user=session_id, stream=False,
        extra_body={"compact_k": compact_k, "memory_expiration": memory_expiration},
    )
    payload = response.model_dump(mode="json")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("News Agent returned no assistant content") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("News Agent returned empty assistant content")

    if "messages" in payload:
        replacement = payload["messages"]
        if (not isinstance(replacement, list) or not replacement
                or any(not isinstance(message, dict)
                       or message.get("role") not in {"system", "developer", "user", "assistant"}
                       or not isinstance(message.get("content"), str)
                       or set(message) - {"role", "content", "name"}
                       or ("name" in message and not isinstance(message["name"], str))
                       for message in replacement)
                or replacement[-1]["role"] != "assistant"
                or not replacement[-1]["content"].strip()):
            raise RuntimeError("News Agent returned an invalid replacement messages history")
        # It already includes this completed turn, either verbatim or summarized.
        next_history = [dict(message) for message in replacement]
    else:
        # Redirects/rejections and older servers may omit the custom history.
        next_history = [*pending, {"role": "assistant", "content": content}]
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise RuntimeError("News Agent returned invalid completion metadata")
    return content, next_history, {
        **metadata, "version": payload.get("version"),
        "compacted": payload.get("compacted", False),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Chat with the governed News Agent")
    parser.add_argument("--url", default=settings.api_url)
    parser.add_argument("--timeout", type=float, default=1200.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    from openai import OpenAI

    args = parse_args(argv)
    settings = load_settings()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("timeout must be finite and greater than 0")
    history: list[dict[str, str]] = []
    base_url = normalize_api_url(args.url, settings.api_url)
    print(f"News session: {settings.mem0_session_id}. Enter exit or quit to finish.")
    with OpenAI(base_url=base_url, api_key="not-needed", timeout=args.timeout, max_retries=0) as client:
        while True:
            try:
                question = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.strip().lower() in {"exit", "quit"}:
                break
            if not question.strip():
                continue
            try:
                content, updated, metadata = send_turn(
                    client, model=settings.public_model_name, history=history,
                    question=question, session_id=settings.mem0_session_id,
                    compact_k=settings.compact_k, memory_expiration=settings.memory_expiration,
                )
            except KeyboardInterrupt:
                print()
                break
            except Exception as exc:
                print(f"News Agent request failed: {exc}")
                continue
            history = updated
            print(f"\nAssistant: {content}\n")
            if metadata.get("compacted"):
                print(f"[History compacted: version {metadata['version']}]\n")


if __name__ == "__main__":
    main()
