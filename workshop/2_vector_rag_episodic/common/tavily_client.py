"""Tavily internet-search client used by the episodic-memory RAG flow."""
from __future__ import annotations

from typing import Any, Dict, List

from .config import Settings


class TavilySearchClient:
    """Run exact-question web searches through the Tavily Python SDK."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.client = client

    def search(self, question: str) -> List[Dict[str, Any]]:
        """Return normalized Tavily results for the exact question."""
        # Check before SDK import, client initialization, or network access.
        if not self.settings.web_search_enabled:
            return []
        if not question.strip():
            raise ValueError("Search question must not be empty")

        if self.client is None:
            from tavily import TavilyClient

            self.client = TavilyClient(api_key=self.settings.tavily_api_key or None)

        response = self.client.search(
            query=question,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
            max_results=self.settings.web_search_max_results,
            timeout=self.settings.web_search_timeout,
        )
        raw_results = response.get("results", []) if isinstance(response, dict) else []

        results: List[Dict[str, Any]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "provider": "tavily",
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "content": str(item.get("content") or ""),
                    "score": item.get("score"),
                }
            )
        return results


__all__ = ["TavilySearchClient"]
