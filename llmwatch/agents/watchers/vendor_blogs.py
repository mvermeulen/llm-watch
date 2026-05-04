"""
Vendor AI blog watchers (feed-first implementation).

Phase 1 quick wins use public RSS/Atom feeds for major vendors to keep
maintenance low and avoid brittle HTML scraping.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from llmwatch.agents.base import BaseAgent, registry

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 20
_DEFAULT_LOOKBACK_DAYS = 14
_DEFAULT_MAX_ITEMS = 30
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class VendorFeedConfig:
    """Configuration for one vendor feed watcher instance."""

    agent_name: str
    label: str
    source_slug: str
    feed_url: str


_VENDOR_FEEDS: tuple[VendorFeedConfig, ...] = (
    VendorFeedConfig(
        agent_name="openai_news_feed",
        label="OpenAI News",
        source_slug="openai_news",
        feed_url="https://openai.com/news/rss.xml",
    ),
    VendorFeedConfig(
        agent_name="google_ai_blog_feed",
        label="Google AI Blog",
        source_slug="google_ai_blog",
        feed_url="https://blog.google/innovation-and-ai/technology/ai/rss/",
    ),
    VendorFeedConfig(
        agent_name="deepmind_blog_feed",
        label="Google DeepMind Blog",
        source_slug="deepmind_blog",
        feed_url="https://deepmind.google/blog/rss.xml",
    ),
    VendorFeedConfig(
        agent_name="microsoft_ai_blog_feed",
        label="Microsoft AI Blog",
        source_slug="microsoft_ai_blog",
        feed_url="https://blogs.microsoft.com/ai/feed/",
    ),
    VendorFeedConfig(
        agent_name="aws_ml_blog_feed",
        label="AWS ML Blog",
        source_slug="aws_ml_blog",
        feed_url="https://aws.amazon.com/blogs/machine-learning/feed/",
    ),
    VendorFeedConfig(
        agent_name="qwen_blog_feed",
        label="Qwen Blog",
        source_slug="qwen_blog",
        feed_url="https://qwenlm.github.io/blog/index.xml",
    ),
)

PHASE1_VENDOR_BLOG_AGENT_NAMES: tuple[str, ...] = tuple(cfg.agent_name for cfg in _VENDOR_FEEDS)

PHASE1_VENDOR_BLOG_ALIASES: dict[str, str] = {
    "openai": "openai_news_feed",
    "google": "google_ai_blog_feed",
    "deepmind": "deepmind_blog_feed",
    "microsoft": "microsoft_ai_blog_feed",
    "aws": "aws_ml_blog_feed",
    "qwen": "qwen_blog_feed",
}


class VendorBlogFeedWatcher(BaseAgent):
    """Generic RSS/Atom watcher for vendor blog/news feeds."""

    category = "watcher"

    def __init__(self, config: VendorFeedConfig) -> None:
        self.config = config
        self.name = config.agent_name

    def run(self, context: dict[str, Any] | None = None):
        context = context or {}
        lookback_days = int(context.get("vendor_blog_lookback_days", _DEFAULT_LOOKBACK_DAYS))
        max_items = int(context.get("vendor_blog_max_items", _DEFAULT_MAX_ITEMS))
        per_feed_limits = context.get("vendor_blog_per_feed_max_items", {})
        if isinstance(per_feed_limits, dict):
            per_feed_value = per_feed_limits.get(self.name)
            if per_feed_value is not None:
                try:
                    max_items = int(per_feed_value)
                except (TypeError, ValueError):
                    logger.warning(
                        "%s: ignoring invalid per-feed max '%s'",
                        self.config.label,
                        per_feed_value,
                    )

        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))

        logger.info("%s: fetching %s", self.config.label, self.config.feed_url)
        try:
            resp = requests.get(
                self.config.feed_url,
                timeout=_REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "llm-watch/0.1 (https://github.com/mvermeulen/llm-watch)"
                },
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            msg = f"{self.config.label} request failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        try:
            entries = _parse_feed_entries(resp.text)
        except ET.ParseError as exc:
            msg = f"{self.config.label} feed parse failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        data: list[dict[str, Any]] = []
        max_items = max(1, max_items)
        for entry in entries:
            if len(data) >= max_items:
                break

            title = _clean_text(entry.get("title", ""))
            url = entry.get("url", "").strip()
            description = _clean_text(entry.get("description", ""))
            published_dt = _parse_feed_datetime(entry.get("published", ""))

            if not title or not url:
                continue
            if published_dt is not None and published_dt < cutoff:
                continue

            categories = [
                _clean_text(cat).lower()
                for cat in entry.get("categories", [])
                if _clean_text(cat)
            ]
            tags = [self.config.source_slug]
            for category in categories[:3]:
                if category not in tags:
                    tags.append(category)

            published_date = (
                published_dt.date().isoformat() if published_dt is not None else ""
            )

            data.append(
                {
                    "model_id": title,
                    "url": url,
                    "description": description,
                    "tags": tags,
                    "source": self.config.source_slug,
                    "published": published_date,
                }
            )

        logger.info("%s: collected %d feed item(s)", self.config.label, len(data))
        return self._result(data=data)


def _clean_text(text: str) -> str:
    plain = _STRIP_TAGS_RE.sub("", html_lib.unescape(text or "")).strip()
    return re.sub(r"\s+", " ", plain)


def _parse_feed_entries(feed_text: str) -> list[dict[str, Any]]:
    """Parse RSS or Atom feed text into normalized entries."""
    root = ET.fromstring(feed_text)
    local_root = _local_name(root.tag)

    if local_root == "feed":
        return _parse_atom_entries(root)
    if local_root == "rss" or root.find("channel") is not None:
        return _parse_rss_entries(root)

    # Fallback: handle feeds that still expose <entry> in mixed docs.
    entry_nodes = [node for node in root.iter() if _local_name(node.tag) == "entry"]
    if entry_nodes:
        return _parse_atom_entries(root)

    return _parse_rss_entries(root)


def _parse_rss_entries(root: ET.Element) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in root.findall("./channel/item"):
        categories = [
            (cat.text or "").strip()
            for cat in item.findall("category")
            if (cat.text or "").strip()
        ]
        entries.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "url": (item.findtext("link") or "").strip(),
                "description": (
                    item.findtext("description")
                    or item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded")
                    or ""
                ).strip(),
                "published": (
                    item.findtext("pubDate")
                    or item.findtext("{http://purl.org/dc/elements/1.1/}date")
                    or ""
                ).strip(),
                "categories": categories,
            }
        )
    return entries


def _parse_atom_entries(root: ET.Element) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    entry_nodes = [node for node in root.iter() if _local_name(node.tag) == "entry"]
    for entry in entry_nodes:
        categories: list[str] = []
        for cat in entry:
            if _local_name(cat.tag) != "category":
                continue
            term = (cat.get("term") or (cat.text or "")).strip()
            if term:
                categories.append(term)

        link = ""
        for link_el in entry:
            if _local_name(link_el.tag) != "link":
                continue
            rel = (link_el.get("rel") or "alternate").strip().lower()
            href = (link_el.get("href") or "").strip()
            if href and rel == "alternate":
                link = href
                break
            if href and not link:
                link = href

        entries.append(
            {
                "title": _child_text(entry, "title"),
                "url": link,
                "description": _child_text(entry, "summary") or _child_text(entry, "content"),
                "published": _child_text(entry, "published") or _child_text(entry, "updated"),
                "categories": categories,
            }
        )
    return entries


def _child_text(node: ET.Element, name: str) -> str:
    for child in node:
        if _local_name(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_feed_datetime(raw: str) -> datetime | None:
    if not raw:
        return None

    value = raw.strip()
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


for _cfg in _VENDOR_FEEDS:
    registry.register(VendorBlogFeedWatcher(_cfg))
