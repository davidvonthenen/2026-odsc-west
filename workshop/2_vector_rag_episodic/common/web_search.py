"""Shared current-web retrieval policy for the episodic RAG application."""
from __future__ import annotations

from typing import Any, Dict, List

from .config import Settings
from .duckduckgo_client import DuckDuckGoSearchClient
from .logging import get_logger
from .tavily_client import TavilySearchClient

LOGGER = get_logger(__name__)


def _run_search(provider: str, question: str, settings: Settings) -> List[Dict[str, Any]]:
    """Run one configured internet provider with the exact question."""
    if provider == "duckduckgo":
        return DuckDuckGoSearchClient(settings).search(question)
    return TavilySearchClient(settings).search(question)


def search_internet(question: str, settings: Settings) -> List[Dict[str, Any]]:
    """Return no fresh results when disabled; otherwise apply primary/fallback policy.

    This switch does not disable recall of web episodes already stored in Mem0.
    Provider SDKs are not initialized when current web retrieval is disabled.
    """
    if not settings.web_search_enabled:
        LOGGER.info("Current web search disabled (WEB_SEARCH_ENABLED=false)")
        return []

    primary = settings.web_search_provider
    fallback = settings.web_search_fallback
    try:
        results = _run_search(primary, question, settings)
        if results:
            return results
        LOGGER.warning("%s returned no internet results", primary)
    except Exception as exc:
        LOGGER.warning("%s internet search failed: %s", primary, exc)

    if fallback == primary:
        return []
    try:
        return _run_search(fallback, question, settings)
    except Exception as exc:
        LOGGER.warning("%s fallback search failed: %s", fallback, exc)
        return []


__all__ = ["search_internet"]
