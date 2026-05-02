"""
Ollama model library watcher.

Scrapes the public Ollama model library page to find recently added or
popular models.  Ollama does not currently expose a stable JSON API for
its library, so we parse the HTML listing.

Library URL: https://ollama.com/library
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests

from llmwatch.agents.base import AgentResult, BaseAgent, registry

logger = logging.getLogger(__name__)

_OLLAMA_LIBRARY_URL = "https://ollama.com/search"
_REQUEST_TIMEOUT = 15


class OllamaModelWatcher(BaseAgent):
    """
    Scrape the Ollama model library for available models.

    Each item in ``AgentResult.data`` is a dict with at least:

    * ``model_id``    – e.g. ``"llama3.2"``
    * ``description`` – short description from the library page
    * ``tags``        – list of tag strings (sizes, variants)
    * ``url``         – canonical Ollama library URL for the model
    * ``source``      – ``"ollama"``
    """

    name = "ollama_models"
    category = "watcher"

    def run(self, context: dict[str, Any] | None = None) -> AgentResult:
        logger.info("OllamaModelWatcher: fetching model library from %s", _OLLAMA_LIBRARY_URL)
        try:
            resp = requests.get(
                _OLLAMA_LIBRARY_URL,
                headers={
                    "User-Agent": "llm-watch/0.1 (https://github.com/mvermeulen/llm-watch)",
                    "HX-Request": "true",
                },
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            msg = f"Ollama library request failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        data = _parse_ollama_library(resp.text)
        logger.info("OllamaModelWatcher: found %d models", len(data))
        return self._result(data=data)


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return _STRIP_TAGS_RE.sub("", html).strip()


def _parse_ollama_library(html: str) -> list[dict[str, Any]]:
    """
    Parse the Ollama /search htmx response and return a list of model dicts.

    The page renders model cards as <a href="/library/<model>"> blocks
    containing the model name in a <span x-test-search-response-title> element,
    a description in a <p> element, capability badges in
    <span x-test-capability> elements, and size badges in
    <span x-test-size> elements.
    """
    data: list[dict[str, Any]] = []
    seen: set[str] = set()

    for block_match in re.finditer(
        r'<a\s[^>]*href="/library/([a-zA-Z0-9][a-zA-Z0-9_\-\.]+)"[^>]*>(.*?)(?=<a\s[^>]*href="/library/|\Z)',
        html,
        re.DOTALL,
    ):
        model_id = block_match.group(1)
        if model_id in seen:
            continue
        seen.add(model_id)

        block = block_match.group(2)

        # Model name from title span
        name_match = re.search(r'<span\s[^>]*x-test-search-response-title[^>]*>(.*?)</span>', block, re.DOTALL)
        name = _strip_tags(name_match.group(1)) if name_match else model_id

        # Description from <p> following the title
        desc_match = re.search(r'<p\s[^>]*>(.*?)</p>', block, re.DOTALL)
        description = _strip_tags(desc_match.group(1)) if desc_match else ""

        # Capability tags (tools, vision, thinking, etc.)
        capabilities = [
            _strip_tags(m) for m in re.findall(
                r'<span\s[^>]*x-test-capability[^>]*>(.*?)</span>', block, re.DOTALL
            )
        ]

        # Size tags (3b, 8b, 70b, etc.)
        sizes = [
            _strip_tags(m) for m in re.findall(
                r'<span\s[^>]*x-test-size[^>]*>(.*?)</span>', block, re.DOTALL
            )
        ]

        tags = capabilities + sizes

        data.append(
            {
                "model_id": name or model_id,
                "description": description[:200],
                "tags": tags,
                "url": f"https://ollama.com/library/{model_id}",
                "source": "ollama",
            }
        )

    return data


# Register a default instance in the global registry.
registry.register(OllamaModelWatcher())
