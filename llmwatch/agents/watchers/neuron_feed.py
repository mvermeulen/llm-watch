"""
The Neuron feed-first watcher.

Uses the public Atom feed to collect recent newsletter and explainer entries
without scraping post bodies.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from llmwatch.agents.base import BaseAgent, registry

logger = logging.getLogger(__name__)

_NEURON_FEED_URL = "https://www.theneuron.ai/feed/"
_REQUEST_TIMEOUT = 20
_DEFAULT_LOOKBACK_DAYS = 7
_DEFAULT_MAX_ITEMS = 40

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")
_ALLOWED_CATEGORY_TERMS = {"newsletter", "explainer-articles"}


class NeuronFeedWatcher(BaseAgent):
    """Collect recent The Neuron feed entries in a feed-only mode."""

    name = "neuron_feed"
    category = "watcher"

    def run(self, context: dict[str, Any] | None = None):
        context = context or {}
        lookback_days = int(context.get("neuron_lookback_days", _DEFAULT_LOOKBACK_DAYS))
        max_items = int(context.get("neuron_max_items", _DEFAULT_MAX_ITEMS))
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))

        logger.info("NeuronFeedWatcher: fetching feed with %d-day lookback", lookback_days)
        try:
            resp = requests.get(
                _NEURON_FEED_URL,
                timeout=_REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "llm-watch/0.1 (https://github.com/mvermeulen/llm-watch)"
                },
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            msg = f"The Neuron feed request failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            msg = f"The Neuron feed parse failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        data: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", _ATOM_NS):
            if len(data) >= max(1, max_items):
                break

            title = _clean_text(entry.findtext("atom:title", default="", namespaces=_ATOM_NS))
            link_el = entry.find("atom:link", _ATOM_NS)
            url = (link_el.get("href", "") if link_el is not None else "").strip()

            summary = _clean_text(
                entry.findtext("atom:summary", default="", namespaces=_ATOM_NS)
            )
            published_raw = (
                entry.findtext("atom:published", default="", namespaces=_ATOM_NS).strip()
            )
            published = _parse_iso_datetime(published_raw)

            category_el = entry.find("atom:category", _ATOM_NS)
            category_term = (
                category_el.get("term", "").strip().lower() if category_el is not None else ""
            )

            if category_term and category_term not in _ALLOWED_CATEGORY_TERMS:
                continue
            if published is None or published < cutoff:
                continue
            if not title or not url:
                continue

            tags = ["neuron"]
            if category_term:
                tags.append(category_term)

            data.append(
                {
                    "model_id": title,
                    "url": url,
                    "description": summary,
                    "tags": tags,
                    "source": "neuron",
                    "published": published.date().isoformat(),
                    "neuron_category": category_term or "unknown",
                }
            )

        logger.info("NeuronFeedWatcher: collected %d feed item(s)", len(data))
        return self._result(data=data)


def _clean_text(text: str) -> str:
    stripped = _STRIP_TAGS_RE.sub("", html_lib.unescape(text or "")).strip()
    return re.sub(r"\s+", " ", stripped)


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


registry.register(NeuronFeedWatcher())
