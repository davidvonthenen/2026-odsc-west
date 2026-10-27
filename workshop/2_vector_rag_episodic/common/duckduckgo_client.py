"""DuckDuckGo backup internet-search client."""
from __future__ import annotations

from typing import Any, Dict, List

from .config import Settings


class DuckDuckGoSearchClient:
    """Run exact-question searches through the duckduckgo-search package."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.client = client

    def search(self, question: str) -> List[Dict[str, Any]]:
        """Return normalized DuckDuckGo results for the exact question."""
        # Check before SDK import, client initialization, or network access.
        if not self.settings.web_search_enabled:
            return []
        if not question.strip():
            raise ValueError("Search question must not be empty")

        if self.client is None:
            from duckduckgo_search import DDGS

            self.client = DDGS(timeout=int(self.settings.web_search_timeout))

        raw_results = self.client.text(
            question,
            region=self.settings.duckduckgo_region,
            safesearch="moderate",
            max_results=self.settings.web_search_max_results,
        )

        results: List[Dict[str, Any]] = []
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "provider": "duckduckgo",
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("href") or item.get("url") or ""),
                    "content": str(item.get("body") or item.get("content") or ""),
                    "score": item.get("score"),
                }
            )
        return results


__all__ = ["DuckDuckGoSearchClient"]
