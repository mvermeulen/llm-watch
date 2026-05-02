"""
HuggingFace trending-model watcher.

Uses the public HuggingFace Hub API to fetch the current list of trending
models.  No authentication is required for this read-only endpoint.

API reference:
  https://huggingface.co/docs/hub/api#get-apiv2models
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests

from llmwatch.agents.base import AgentResult, BaseAgent, registry

logger = logging.getLogger(__name__)

_HF_API_URL = "https://huggingface.co/api/models"
_DEFAULT_LIMIT = 20
_REQUEST_TIMEOUT = 15


class HuggingFaceTrendingWatcher(BaseAgent):
    """
    Fetch the top trending models from the HuggingFace Hub.

    Each item in ``AgentResult.data`` is a dict with at least:

    * ``model_id``   – e.g. ``"mistralai/Mistral-7B-Instruct-v0.2"``
    * ``author``     – model owner
    * ``downloads``  – download count (last 30 days)
    * ``likes``      – number of likes
    * ``tags``       – list of pipeline-tag / library strings
    * ``url``        – canonical HuggingFace URL for the model
    * ``description``– first sentence of the model card (if available)
    """

    name = "huggingface_trending"
    category = "watcher"

    def __init__(self, limit: int = _DEFAULT_LIMIT) -> None:
        self.limit = limit

    def run(self, context: dict[str, Any] | None = None) -> AgentResult:
        logger.info("HuggingFaceTrendingWatcher: fetching top %d trending models", self.limit)
        try:
            resp = requests.get(
                _HF_API_URL,
                params={"sort": "trending", "limit": self.limit, "full": "False"},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            msg = f"HuggingFace API request failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        models = resp.json()
        data: list[dict[str, Any]] = []
        new_sources: list[str] = []

        for m in models:
            model_id = m.get("id", "")
            tags = m.get("tags", [])
            author = m.get("author", model_id.split("/")[0] if "/" in model_id else "")
            url = f"https://huggingface.co/{model_id}"

            # Extract any external URLs from the card data as potential new sources
            card_data = m.get("cardData", {}) or {}
            for link in _extract_urls_from_card(card_data):
                if link not in new_sources:
                    new_sources.append(link)

            data.append(
                {
                    "model_id": model_id,
                    "author": author,
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "tags": tags,
                    "url": url,
                    "description": _safe_description(m),
                    "source": "huggingface",
                }
            )

        logger.info("HuggingFaceTrendingWatcher: retrieved %d models", len(data))
        return self._result(data=data, new_sources=new_sources)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def _extract_urls_from_card(card_data: dict) -> list[str]:
    """Walk a card-data dict and return any URL-like strings found."""
    found: list[str] = []
    for value in card_data.values():
        if isinstance(value, str):
            found.extend(_URL_RE.findall(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    found.extend(_URL_RE.findall(item))
    return found


def _safe_description(model_meta: dict) -> str:
    """Return a short description from available model metadata."""
    # The /api/models endpoint sometimes returns a 'description' field
    desc = model_meta.get("description") or ""
    if desc:
        # Truncate to first sentence or 200 chars
        end = desc.find(". ")
        if 0 < end < 200:
            return desc[: end + 1]
        return desc[:200]
    return ""


# Register a default instance in the global registry.
registry.register(HuggingFaceTrendingWatcher())
