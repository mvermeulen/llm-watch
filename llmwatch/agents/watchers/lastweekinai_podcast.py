"""
Last Week in AI podcast watcher.

Pulls recent episodes from the public RSS feed and extracts the summary links so
we can review the referenced sources without listening to the full audio.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import requests

from llmwatch.agents.base import BaseAgent, registry

logger = logging.getLogger(__name__)

_FEED_URL = "https://lastweekin.ai/feed"
_REQUEST_TIMEOUT = 20
_LINK_REQUEST_TIMEOUT = 8
_DEFAULT_LOOKBACK_DAYS = 7
_DEFAULT_MAX_LINKS_PER_EPISODE = 25

_A_TAG_RE = re.compile(
    r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")
_PROFILE_PATH_RE = re.compile(r"^/[A-Za-z0-9_\.-]+/?$")

_CONTENT_ENCODED_TAG = "{http://purl.org/rss/1.0/modules/content/}encoded"

_LOW_SIGNAL_DOMAINS = {
    "x.com",
    "twitter.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "youtube-nocookie.com",
    "youtu.be",
}

_ARTICLE_HOST_ALLOWLIST = {
    "arxiv.org",
    "openai.com",
    "anthropic.com",
    "huggingface.co",
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "reuters.com",
    "bloomberg.com",
    "nature.com",
    "science.org",
}

_ARTICLE_PATH_HINTS = (
    "/abs/",
    "/pdf/",
    "/news/",
    "/article/",
    "/articles/",
    "/blog/",
    "/research/",
    "/story/",
    "/story",
)

_ARTICLE_TEXT_HINTS = (
    "arxiv",
    "paper",
    "research",
    "study",
    "report",
    "blog",
    "news",
)


class LastWeekInAIPodcastWatcher(BaseAgent):
    """
    Extract recent Last Week in AI podcast summaries and referenced links.

    Returns one item per episode summary and one item per referenced external
    source URL.
    """

    name = "lastweekinai_podcast"
    category = "watcher"

    def run(self, context: dict[str, Any] | None = None):
        context = context or {}
        lookback_days = int(context.get("lwiai_lookback_days", _DEFAULT_LOOKBACK_DAYS))
        max_links_per_episode = int(
            context.get("lwiai_max_links_per_episode", _DEFAULT_MAX_LINKS_PER_EPISODE)
        )

        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))
        logger.info(
            "LastWeekInAIPodcastWatcher: fetching feed with %d-day lookback", lookback_days
        )

        try:
            resp = requests.get(
                _FEED_URL,
                timeout=_REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "llm-watch/0.1 (https://github.com/mvermeulen/llm-watch)"
                },
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            msg = f"Last Week in AI feed request failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            msg = f"Last Week in AI feed parse failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        data: list[dict[str, Any]] = []
        new_sources: list[str] = []

        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            if not _is_podcast_title(title):
                continue

            pub_date_raw = (item.findtext("pubDate") or "").strip()
            pub_date = _parse_pub_date(pub_date_raw)
            if pub_date is None:
                continue
            if pub_date < cutoff:
                continue

            episode_url = (item.findtext("link") or "").strip()
            description = _clean_text(item.findtext("description") or "")
            content_html = item.findtext(_CONTENT_ENCODED_TAG) or ""

            data.append(
                {
                    "model_id": title,
                    "url": episode_url,
                    "description": description,
                    "tags": ["podcast_summary"],
                    "source": "lastweekinai_podcast",
                    "episode_title": title,
                    "episode_url": episode_url,
                    "published": pub_date.date().isoformat(),
                }
            )

            link_items = _extract_episode_links(
                content_html=content_html,
                episode_title=title,
                episode_url=episode_url,
                published=pub_date.date().isoformat(),
                max_links=max_links_per_episode,
            )
            data.extend(link_items)
            for link_item in link_items:
                link_url = link_item.get("url", "")
                if link_url and link_url not in new_sources:
                    new_sources.append(link_url)

        logger.info(
            "LastWeekInAIPodcastWatcher: collected %d data item(s) from recent episodes",
            len(data),
        )
        return self._result(data=data, new_sources=new_sources)


def _is_podcast_title(title: str) -> bool:
    lower = title.lower()
    return "podcast" in lower or lower.startswith("lwiai")


def _parse_pub_date(pub_date: str) -> datetime | None:
    if not pub_date:
        return None
    try:
        parsed = parsedate_to_datetime(pub_date)
    except (TypeError, ValueError):
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_episode_links(
    content_html: str,
    episode_title: str,
    episode_url: str,
    published: str,
    max_links: int,
) -> list[dict[str, Any]]:
    """Extract external links and enrich each with a fetched page title."""
    links: list[dict[str, Any]] = []
    seen: set[str] = set()

    for match in _A_TAG_RE.finditer(content_html):
        if len(links) >= max(1, max_links):
            break

        href = html_lib.unescape((match.group(1) or "").strip())
        text = _clean_text(match.group(2) or "")
        if not href.startswith("http"):
            continue
        if _skip_link(href):
            continue
        if not _is_quality_article_link(href, text):
            continue
        if href in seen:
            continue

        seen.add(href)
        resolved_url, resolved_title = _lookup_link(href)
        final_url = resolved_url or href
        if not _is_quality_article_link(final_url, text or resolved_title):
            continue
        display_title = text or resolved_title or _domain(final_url)

        links.append(
            {
                "model_id": display_title,
                "url": final_url,
                "description": f"Referenced in {episode_title}",
                "tags": ["podcast_link", _domain(final_url)],
                "source": "lastweekinai_podcast",
                "episode_title": episode_title,
                "episode_url": episode_url,
                "published": published,
                "referenced_title": text,
                "resolved_page_title": resolved_title,
            }
        )

    return links


def _lookup_link(url: str) -> tuple[str, str]:
    """Return (final_url, title) for a referenced URL when possible."""
    try:
        resp = requests.get(
            url,
            timeout=_LINK_REQUEST_TIMEOUT,
            allow_redirects=True,
            headers={
                "User-Agent": "llm-watch/0.1 (https://github.com/mvermeulen/llm-watch)"
            },
        )
        resp.raise_for_status()
    except requests.RequestException:
        return "", ""

    title_match = _TITLE_RE.search(resp.text)
    title = _clean_text(title_match.group(1)) if title_match else ""
    return (resp.url or url, title)


def _skip_link(url: str) -> bool:
    """Filter out non-news links from feed content."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().lstrip("www.")
    if host in {"lastweekin.ai", "substack.com", "api.substack.com", "youtube.com", "youtube-nocookie.com"}:
        return True
    return url.startswith("mailto:")


def _is_quality_article_link(url: str, anchor_text: str = "") -> bool:
    """Return True for likely article/research links and False for low-signal links."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().lstrip("www.")
    path = parsed.path or ""
    text = (anchor_text or "").strip().lower()

    if not host:
        return False

    if host in _LOW_SIGNAL_DOMAINS:
        return False

    # Drop obvious profile pages.
    if host in {"x.com", "twitter.com", "linkedin.com"} and _PROFILE_PATH_RE.match(path):
        return False

    if host in _ARTICLE_HOST_ALLOWLIST and path and path != "/":
        return True

    if any(hint in path.lower() for hint in _ARTICLE_PATH_HINTS):
        return True

    if re.search(r"/20\d\d/\d\d/", path):
        return True

    if any(hint in text for hint in _ARTICLE_TEXT_HINTS) and len(path) > 8 and path != "/":
        return True

    # Fallback: keep deep-ish paths with slugs, skip root pages.
    return path.count("/") >= 2 and len(path) >= 15


def _clean_text(text: str) -> str:
    stripped = _STRIP_TAGS_RE.sub("", html_lib.unescape(text or "")).strip()
    return re.sub(r"\s+", " ", stripped)


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().lstrip("www.")


registry.register(LastWeekInAIPodcastWatcher())