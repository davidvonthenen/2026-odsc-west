"""Interactive REST client using the OpenAI SDK and a stable Mem0 session."""
from __future__ import annotations

import sys
import time
from collections.abc import Sequence

from src.common.config import load_settings
from src.query import normalize_api_url


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    from openai import OpenAI

    parser = argparse.ArgumentParser(description="Chat with the memory-enabled News Agent")
    parser.add_argument("--url", default=None, help="Agent origin or /v1 base URL")
    args = parser.parse_args(argv)
    settings = load_settings()
    base_url = normalize_api_url(args.url or settings.api_url, settings.api_url)
    print("News Agent")
    print(f"Session: {settings.mem0_session_id}. Ask about technology news. Type exit to stop.")
    messages: list[dict[str, str]] = []
    # Retrying a stateful POST after a lost response could duplicate a memory turn.
    # No hosted service is contacted: base_url points to the configured agent.
    with OpenAI(base_url=base_url, api_key="not-needed", timeout=1200.0,
                max_retries=0) as client:
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            pending = [*messages, {"role": "user", "content": question}]
            started = time.perf_counter()
            try:
                completion = client.chat.completions.create(
                    model=settings.public_model_name,
                    messages=pending,
                    user=settings.mem0_session_id,
                    stream=False,
                )
                answer = completion.choices[0].message.content
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("News Agent returned no assistant content")
            except KeyboardInterrupt:
                print("\nRequest interrupted. The server may still complete and save this turn.", file=sys.stderr)
                break
            except Exception as exc:
                print(f"News Agent request failed: {exc}", file=sys.stderr)
                print("The request was not retried; its memory writes may already have completed.", file=sys.stderr)
                continue
            messages = [*pending, {"role": "assistant", "content": answer}]
            # The display transcript is not a memory import. The server resumes
            # the authoritative saved conversation using the stable session ID.
            print(f"\nNews Agent:\n{answer}")
            print(f"\nResponse time: {time.perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()
