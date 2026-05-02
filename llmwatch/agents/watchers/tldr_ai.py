"""
TLDR AI newsletter watcher.

Fetches the current day's TLDR AI newsletter and extracts article headlines,
URLs, and descriptions grouped by section.

Newsletter URL pattern: https://tldr.tech/ai/YYYY-MM-DD
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from datetime import date, timedelta
from typing import Any

import requests

from llmwatch.agents.base import AgentResult, BaseAgent, registry

logger = logging.getLogger(__name__)

_TLDR_BASE_URL = "https://tldr.tech/ai"
_REQUEST_TIMEOUT = 15
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")

# Sponsor link domains to skip
_SPONSOR_DOMAINS = frozenset(
    ["ref.wisprflow.ai", "jobs.ashbyhq.com", "advertise.tldr.tech"]
)


def _strip_tags(text: str) -> str:
    return html_lib.unescape(_STRIP_TAGS_RE.sub("", text)).strip()


class TLDRAIWatcher(BaseAgent):
    """
    Fetch today's TLDR AI newsletter and return each article as a data item.

    Each item in ``AgentResult.data`` contains:

    * ``model_id``    – article title (used by the reporter as the display name)
    * ``url``         – article URL (sponsor UTM params stripped)
    * ``description`` – article summary text
    * ``tags``        – list containing the newsletter section name
    * ``source``      – ``"tldr_ai"``
    """

    name = "tldr_ai"
    category = "watcher"

    def run(self, context: dict[str, Any] | None = None) -> AgentResult:
        # Try today, then fall back up to 6 days until we find a published edition.
        for days_back in range(7):
            edition_date = date.today() - timedelta(days=days_back)
            url = f"{_TLDR_BASE_URL}/{edition_date.isoformat()}"
            logger.info("TLDRAIWatcher: fetching newsletter from %s", url)
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": "llm-watch/0.1 (https://github.com/mvermeulen/llm-watch)"},
                    timeout=_REQUEST_TIMEOUT,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                msg = f"TLDR AI request failed: {exc}"
                logger.error(msg)
                return self._result(errors=[msg])

            # A redirect means the edition isn't published yet; try previous day.
            if resp.is_redirect:
                logger.info("TLDRAIWatcher: %s not yet published, trying previous day", edition_date)
                continue

            if not resp.ok:
                msg = f"TLDR AI request failed: {resp.status_code} for {url}"
                logger.error(msg)
                return self._result(errors=[msg])

            break
        else:
            msg = "TLDRAIWatcher: could not find a published edition within the last 7 days"
            logger.error(msg)
            return self._result(errors=[msg])

        data = _parse_tldr_newsletter(resp.text)
        logger.info("TLDRAIWatcher: found %d articles in %s edition", len(data), edition_date)
        return self._result(data=data)


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def _is_sponsor(url: str) -> bool:
    """Return True if the URL belongs to a known sponsor/ad domain."""
    for domain in _SPONSOR_DOMAINS:
        if domain in url:
            return True
    return False


def _clean_url(url: str) -> str:
    """Remove utm_source tracking param from the URL."""
    return re.sub(r"[?&]utm_source=[^&]+", lambda m: "" if m.group(0).startswith("?") else "", url).rstrip("?&")


def _parse_tldr_newsletter(html: str) -> list[dict[str, Any]]:
    """
    Parse the TLDR AI newsletter HTML.

    The page is structured as ``<section>`` blocks.  Each section has a
    ``<header>`` containing the section name in an ``<h3>`` element, followed
    by ``<article>`` elements.  Each article has:

    * ``<a class="font-bold" href="..."><h3>Title (N minute read)</h3></a>``
    * ``<div class="newsletter-html">Description text …</div>``
    """
    data: list[dict[str, Any]] = []

    for section_match in re.finditer(
        r"<section>(.*?)(?=<section>|\Z)", html, re.DOTALL
    ):
        section_html = section_match.group(1)

        # Extract section name from <h3> inside <header>
        header_match = re.search(r"<header[^>]*>.*?<h3[^>]*>(.*?)</h3>", section_html, re.DOTALL)
        section_name = _strip_tags(header_match.group(1)) if header_match else "General"

        for article_match in re.finditer(r"<article[^>]*>(.*?)</article>", section_html, re.DOTALL):
            article_html = article_match.group(1)

            # Title and URL from the bold anchor
            link_match = re.search(
                r'<a\s[^>]*class="font-bold"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                article_html,
                re.DOTALL,
            )
            if not link_match:
                continue

            article_url = html_lib.unescape(link_match.group(1))
            if _is_sponsor(article_url):
                continue

            title = _strip_tags(link_match.group(2))
            # Strip trailing read-time annotation, e.g. " (4 minute read)"
            title = re.sub(r"\s*\(\d+\s+minute\s+read\)\s*$", "", title).strip()

            # Description from newsletter-html div
            desc_match = re.search(
                r'<div[^>]*class="newsletter-html"[^>]*>(.*?)</div>',
                article_html,
                re.DOTALL,
            )
            description = _strip_tags(desc_match.group(1))[:300] if desc_match else ""

            data.append(
                {
                    "model_id": title,
                    "url": _clean_url(article_url),
                    "description": description,
                    "tags": [section_name],
                    "source": "tldr_ai",
                }
            )

    return data


# Register a default instance in the global registry.
registry.register(TLDRAIWatcher())
