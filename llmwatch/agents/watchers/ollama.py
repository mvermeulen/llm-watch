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

_OLLAMA_LIBRARY_URL = "https://ollama.com/library"
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
                headers={"User-Agent": "llm-watch/0.1 (https://github.com/mvermeulen/llm-watch)"},
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

# These patterns are intentionally loose so they survive minor HTML changes.
_MODEL_BLOCK_RE = re.compile(
    r'href="/([a-zA-Z0-9_\-\.]+)"[^>]*>.*?'
    r'<(?:h2|span)[^>]*>([^<]+)<',
    re.DOTALL,
)
_DESC_RE = re.compile(r'<p[^>]*class="[^"]*(?:description|text)[^"]*"[^>]*>(.*?)</p>', re.DOTALL)
_TAG_RE = re.compile(r'<span[^>]*>([\d\.]+[BbKk]?(?:\s*params)?)</span>')
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return _STRIP_TAGS_RE.sub("", html).strip()


def _parse_ollama_library(html: str) -> list[dict[str, Any]]:
    """
    Parse the Ollama library HTML and return a list of model dicts.

    We look for anchor tags that point to model pages and try to extract
    the model name and description from the surrounding markup.
    """
    data: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Find all <li> or <article> blocks that contain a model link
    # The page uses Next.js so the HTML structure may vary; we use a broad
    # anchor-based scan.
    for match in re.finditer(
        r'href="/([a-zA-Z0-9][a-zA-Z0-9_\-\.]{1,60})"[^>]*>(.*?)(?=href="/[a-zA-Z]|\Z)',
        html,
        re.DOTALL,
    ):
        model_id = match.group(1)
        block = match.group(2)

        # Skip navigation / non-model paths
        if any(
            model_id.startswith(p)
            for p in ("search", "library", "blog", "docs", "about", "download", "api")
        ):
            continue
        if model_id in seen:
            continue
        seen.add(model_id)

        # Extract description
        desc_match = _DESC_RE.search(block)
        description = _strip_tags(desc_match.group(1)) if desc_match else ""

        # Extract size tags
        tags = [t.strip() for t in _TAG_RE.findall(block)]

        # If we found nothing useful, try a plain-text extraction
        if not description:
            plain = _strip_tags(block)
            lines = [l.strip() for l in plain.splitlines() if l.strip()]
            description = lines[0] if lines else ""

        data.append(
            {
                "model_id": model_id,
                "description": description[:200],
                "tags": tags,
                "url": f"https://ollama.com/{model_id}",
                "source": "ollama",
            }
        )

    return data


# Register a default instance in the global registry.
registry.register(OllamaModelWatcher())
