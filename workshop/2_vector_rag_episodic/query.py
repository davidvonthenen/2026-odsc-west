"""Vector RAG client with web retrieval and Mem0 episodic memory."""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Sequence

from common.common import ask, load_llm
from common.config import load_settings
from common.conversation import ConversationManager
from common.logging import get_logger
from common.mem0_client import Mem0Client
from common.web_search import search_internet as _search_internet

LOGGER = get_logger(__name__)


def _score(value: Any) -> float:
    """Convert an optional score to a printable float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _print_hits(hits: List[dict]) -> None:
    """Print compact metadata for the retrieved vector chunks."""
    print("\nRAG Hits:")
    if not hits:
        print("  (none)")
        return
    for hit in hits:
        print(
            f"  - {hit.get('title', 'unknown')} | "
            f"{hit.get('category', 'unknown')} | "
            f"score={hit.get('score', 0.0):.4f}"
        )


def _print_web_results(results: Sequence[Dict[str, Any]]) -> None:
    """Print compact metadata for current internet search results."""
    print("\nInternet Hits:")
    if not results:
        print("  (none)")
        return
    for result in results:
        print(
            f"  - {result.get('provider', 'unknown')} | "
            f"{result.get('title', 'unknown')} | {result.get('url', '')}"
        )


def _print_memories(memories: Sequence[Dict[str, Any]]) -> None:
    """Print compact metadata for recalled Mem0 entries."""
    print("\nRecalled Episodic Memories:")
    if not memories:
        print("  (none)")
        return
    for memory in memories:
        metadata = memory.get("metadata") or {}
        text = str(memory.get("memory") or "").replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        print(
            f"  - {metadata.get('memory_kind', 'unknown')} | "
            f"score={_score(memory.get('score')):.4f} | {text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Ask one question or run the original demo questions with episodic memory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", required=True, type=str, help="Ask a single question and exit")
    args = parser.parse_args(argv)

    settings = load_settings()
    llm = load_llm(settings)
    memory = Mem0Client(settings)
    conversation = ConversationManager(settings, memory)

    try:
        conversation_history = conversation.load_complete_conversation()
        if conversation_history:
            LOGGER.info(
                "Loaded %s messages from Mem0 run '%s'",
                len(conversation_history),
                settings.mem0_complete_run_id,
            )

        # ask the Q
        started = time.time()
        question = args.question

        recalled_memories = conversation.retrieve_topic_memories(question)
        internet_results = _search_internet(question, settings)
        added_web_episodes = (
            conversation.save_internet_results(question, internet_results)
            if internet_results else 0
        )

        answer, hits = ask(
            llm,
            question,
            settings=settings,
            top_k=settings.rag_top_k,
            num_candidates=settings.rag_num_candidates,
            internet_results=internet_results,
            episodic_memories=recalled_memories,
            conversation_history=conversation_history,
        )

        conversation_history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )

        saved_conversation_chunks = conversation.save_complete_conversation(
            conversation_history
        )
        synthesized_memories = conversation.save_synthesized_turn(
            question, answer
        )

        elapsed = time.time() - started
        print("\n" + "=" * 88)
        print(f"QUESTION: {question}")
        print("=" * 88)
        print(f"\nANSWER: {answer}")
        _print_hits(hits)
        _print_web_results(internet_results)
        _print_memories(recalled_memories)
        print("=" * 88)
        print(
            f"\nQuery time: {elapsed:.2f}s   "
            "(OpenSearch vector k-NN + web + Mem0 + LLM service)"
        )
        print(f"Docs provided to LLM: {len(hits)}")
        print(f"Current web search enabled: {settings.web_search_enabled}")
        print(f"Internet results provided to LLM: {len(internet_results)}")
        print(f"New internet episodes saved: {added_web_episodes}")
        print(f"Episodic memories recalled: {len(recalled_memories)}")
        print(f"Conversation chunks saved: {saved_conversation_chunks}")
        print(f"Synthesized turn writes: {synthesized_memories}")
        print(f"Conversation messages persisted: {len(conversation_history)}\n")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
