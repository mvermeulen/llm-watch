"""
Hugging Face trending papers watcher.

Uses the public Hugging Face Papers API to fetch currently trending papers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from llmwatch.agents.base import BaseAgent, registry

logger = logging.getLogger(__name__)

_HF_DAILY_PAPERS_API_URL = "https://huggingface.co/api/daily_papers"
_REQUEST_TIMEOUT = 20
_DEFAULT_LIMIT = 20
_DEFAULT_LOOKBACK_DAYS = 7


class HuggingFaceTrendingPapersWatcher(BaseAgent):
    """Fetch trending papers from Hugging Face daily papers feed."""

    name = "huggingface_trending_papers"
    category = "watcher"

    def run(self, context: dict[str, Any] | None = None):
        context = context or {}
        limit = int(context.get("hf_papers_limit", _DEFAULT_LIMIT))
        lookback_days = int(context.get("hf_papers_lookback_days", _DEFAULT_LOOKBACK_DAYS))
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))

        logger.info(
            "HuggingFaceTrendingPapersWatcher: fetching top %d papers (%d-day lookback)",
            limit,
            lookback_days,
        )

        try:
            resp = requests.get(
                _HF_DAILY_PAPERS_API_URL,
                params={"limit": max(1, limit)},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            msg = f"Hugging Face papers API request failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        items = resp.json()
        data: list[dict[str, Any]] = []

        for item in items:
            paper = item.get("paper", {}) or {}
            paper_id = str(paper.get("id", "")).strip()
            title = str(item.get("title") or paper.get("title") or "").strip()
            summary = str(item.get("summary") or paper.get("summary") or "").strip()

            published = _parse_iso_datetime(item.get("publishedAt"))
            if published is None or published < cutoff:
                continue

            authors = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
            upvotes = int((paper.get("upvotes") if isinstance(paper, dict) else 0) or 0)

            paper_url = f"https://huggingface.co/papers/{paper_id}" if paper_id else ""
            tags = ["paper", "huggingface"]
            if item.get("organization"):
                tags.append(str(item.get("organization")).lower())

            data.append(
                {
                    "model_id": title or paper_id or "Untitled paper",
                    "url": paper_url,
                    "description": summary[:300],
                    "authors": ", ".join(authors[:8]),
                    "tags": tags,
                    "source": "huggingface_papers",
                    "paper_id": paper_id,
                    "upvotes": upvotes,
                    "published": published.date().isoformat(),
                }
            )

        logger.info("HuggingFaceTrendingPapersWatcher: retrieved %d papers", len(data))
        return self._result(data=data)


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


registry.register(HuggingFaceTrendingPapersWatcher())
