"""
TLDR AI newsletter watcher.

Fetches the current day's TLDR AI newsletter and extracts article headlines,
URLs, and descriptions grouped by section.

Newsletter URL pattern: https://tldr.tech/ai/YYYY-MM-DD
"""

from __future__ import annotations

import json
import html as html_lib
import logging
import os
import re
from datetime import date, timedelta
from typing import Any

import requests

from llmwatch.agents.base import AgentResult, BaseAgent, registry

logger = logging.getLogger(__name__)

_TLDR_BASE_URL = "https://tldr.tech/ai"
_REQUEST_TIMEOUT = 15
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")
_OLLAMA_API_URL = os.getenv("LLMWATCH_OLLAMA_API_URL", "http://localhost:11434/api/generate")
_OLLAMA_FILTER_MODEL = os.getenv("LLMWATCH_TLDR_FILTER_MODEL", "llama3.2:3b")
_OLLAMA_REQUEST_TIMEOUT = 25
from llmwatch.cache import get_cache_dir as _get_cache_dir
_TLDR_CACHE_PATH = os.path.join(_get_cache_dir(), "tldr_items.json")
_TLDR_HISTORY_DAYS = int(os.getenv("LLMWATCH_TLDR_HISTORY_DAYS", "14"))

# Sponsor link domains to skip
_SPONSOR_DOMAINS = frozenset(
    ["ref.wisprflow.ai", "jobs.ashbyhq.com", "advertise.tldr.tech"]
)

# Keywords that indicate LLM/model-relevant content (whitelist)
_LLM_KEYWORDS = frozenset(
    [
        "model", "llm", "language model", "gpt", "claude", "gemini", "llama",
        "mistral", "qwen", "deepseek", "ollama", "huggingface",
        "launch", "release", "announcement", "new", "checkpoint",
        "training", "fine-tune", "finetune", "reasoning", "agent",
        "inference", "serving", "inference engine", "vllm", "ollama",
        "open source", "open-source", "weights", "parameters",
    ]
)

# Keywords that indicate off-topic content (blacklist)
_EXCLUDE_KEYWORDS = frozenset(
    [
        "hiring", "job", "jobs", "career", "recruiting",
        "acquisition", "acquired", "merger",
        "ipo", "valuation", "funding round",
        "privacy", "regulation", "policy", "gdpr",
        "deepfake", "misinformation", "election",
    ]
)

_CLASS_TRENDING = "trending_new_models"
_CLASS_MODEL_ANALYSIS = "model_analysis"
_CLASS_OTHER = "other"


def _strip_tags(text: str) -> str:
    return html_lib.unescape(_STRIP_TAGS_RE.sub("", text)).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _is_ollama_filter_enabled() -> bool:
    return _env_bool("LLMWATCH_TLDR_OLLAMA_FILTER", True)


def _is_relevant_to_llm_models(title: str, description: str) -> bool:
    """
    Filter articles to keep only those directly relevant to LLM models.
    
    Uses a combination of:
    1. Blacklist: if title/description contains exclude keywords, filter out
    2. Whitelist: if title/description contains LLM keywords, keep it
    3. Default: keep items that aren't clearly off-topic
    
    Returns True if the item should be included.
    """
    text = (title + " " + description).lower()
    
    # Hard exclude if blacklist keywords present
    for keyword in _EXCLUDE_KEYWORDS:
        if keyword in text:
            logger.debug("TLDR filtering out (blacklist): %s", title[:60])
            return False
    
    # Keep if whitelist keywords present
    for keyword in _LLM_KEYWORDS:
        if keyword in text:
            return True
    
    # Default: keep items that don't have obvious exclusion keywords
    # (this is lenient to avoid false negatives)
    logger.debug("TLDR keeping (no exclusion keywords): %s", title[:60])
    return True


def _classify_with_ollama(title: str, description: str, section: str) -> tuple[bool, str] | None:
    """
    Classify TLDR content with a local Ollama model.

    Returns:
        (include_in_trending, category) on success, otherwise None.
    """
    prompt = (
        "You are classifying TLDR AI newsletter items for an LLM model watch report. "
        "Return ONLY minified JSON with keys include_in_trending (boolean) and "
        "category (one of: trending_new_models, model_analysis, other). "
        "Use trending_new_models for direct model launches/releases/checkpoints. "
        "Use model_analysis for deep dives about model behavior, inference, or performance. "
        "Use other for business/news not primarily about model technology.\n\n"
        f"Section: {section}\n"
        f"Title: {title}\n"
        f"Description: {description}\n"
    )
    payload = {
        "model": _OLLAMA_FILTER_MODEL,
        "stream": False,
        "format": "json",
        "prompt": prompt,
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(
            _OLLAMA_API_URL,
            json=payload,
            timeout=_OLLAMA_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "").strip()
        parsed = json.loads(response_text)
        include = bool(parsed.get("include_in_trending", False))
        category = str(parsed.get("category", _CLASS_OTHER)).strip().lower()
        if category not in {_CLASS_TRENDING, _CLASS_MODEL_ANALYSIS, _CLASS_OTHER}:
            category = _CLASS_OTHER
        return include, category
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        logger.debug("TLDR Ollama filter unavailable/invalid output: %s", exc)
        return None


def _classify_item(title: str, description: str, section: str) -> tuple[bool, str]:
    """Hybrid classifier: local Ollama first, rules fallback."""
    if _is_ollama_filter_enabled():
        classified = _classify_with_ollama(title, description, section)
        if classified is not None:
            return classified

    include = _is_relevant_to_llm_models(title, description)
    return include, _CLASS_TRENDING if include else _CLASS_OTHER


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
        context = context or {}

        # Check if date_range is provided (for backfilling multiple days)
        date_range = context.get("date_range")
        all_data = []

        if date_range:
            # Fetch data for each date in the range
            start_date, end_date = date_range
            current_date = start_date
            while current_date <= end_date:
                edition_data = self._fetch_single_edition(current_date)
                if edition_data:
                    all_data.extend(edition_data)
                    logger.info(
                        "TLDRAIWatcher: found %d articles in %s edition",
                        len(edition_data),
                        current_date.isoformat(),
                    )
                current_date += timedelta(days=1)

            if not all_data:
                msg = f"TLDRAIWatcher: no published editions found in date range {start_date} to {end_date}"
                logger.warning(msg)

            merged = _merge_with_cached_tldr_items(all_data)
            logger.info(
                "TLDRAIWatcher: found %d articles across date range (%d cached total)",
                len(all_data),
                len(merged),
            )
            return self._result(data=merged)

        # Legacy behavior: Try today, then fall back up to 6 days until we find a published edition.
        edition_data = None
        edition_date = None
        for days_back in range(7):
            edition_date = date.today() - timedelta(days=days_back)
            edition_data = self._fetch_single_edition(edition_date)
            if edition_data:
                break
        else:
            msg = "TLDRAIWatcher: could not find a published edition within the last 7 days"
            logger.error(msg)
            return self._result(errors=[msg])

        merged = _merge_with_cached_tldr_items(edition_data)
        logger.info(
            "TLDRAIWatcher: found %d articles in %s edition (%d cached total)",
            len(edition_data),
            edition_date,
            len(merged),
        )
        return self._result(data=merged)

    def _fetch_single_edition(self, edition_date: date) -> list[dict[str, Any]] | None:
        """
        Fetch a single TLDR edition for the given date.

        Args:
            edition_date: The date of the edition to fetch

        Returns:
            Parsed data if successful, None otherwise.
        """
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
            logger.debug("TLDRAIWatcher: request failed for %s: %s", edition_date, exc)
            return None

        # Check for redirect status codes (300-399)
        if 300 <= resp.status_code < 400:
            logger.debug("TLDRAIWatcher: %s not yet published (status %s)", edition_date, resp.status_code)
            return None

        if not resp.ok:
            logger.debug("TLDRAIWatcher: %s request failed: %s", edition_date, resp.status_code)
            return None

        return _parse_tldr_newsletter(resp.text, edition_date.isoformat())


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


def _load_cached_tldr_items() -> list[dict[str, Any]]:
    if not os.path.exists(_TLDR_CACHE_PATH):
        return []
    try:
        with open(_TLDR_CACHE_PATH, "r", encoding="utf-8") as fh:
            items = json.load(fh)
        return items if isinstance(items, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("TLDRAIWatcher: failed to load cache: %s", exc)
        return []


def _save_cached_tldr_items(items: list[dict[str, Any]]) -> None:
    try:
        os.makedirs(os.path.dirname(_TLDR_CACHE_PATH), exist_ok=True)
        with open(_TLDR_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(items, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("TLDRAIWatcher: failed to save cache: %s", exc)


def _merge_with_cached_tldr_items(new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge new TLDR items into cache, de-duplicated and date-pruned."""
    cached = _load_cached_tldr_items()
    combined = [*cached, *new_items]

    cutoff = date.today() - timedelta(days=max(_TLDR_HISTORY_DAYS, 1))
    deduped: dict[str, dict[str, Any]] = {}
    for item in combined:
        edition = str(item.get("edition_date", "")).strip()
        try:
            item_date = date.fromisoformat(edition) if edition else date.today()
        except ValueError:
            item_date = date.today()

        if item_date < cutoff:
            continue

        key = item.get("url") or f"{item.get('model_id', '')}|{edition}"
        if not key:
            continue

        existing = deduped.get(key)
        if existing is None or str(item.get("edition_date", "")) > str(existing.get("edition_date", "")):
            deduped[key] = item

    merged = sorted(
        deduped.values(),
        key=lambda i: (i.get("edition_date", ""), i.get("model_id", "")),
        reverse=True,
    )
    _save_cached_tldr_items(merged)
    return merged


def _parse_tldr_newsletter(html: str, edition_date: str | None = None) -> list[dict[str, Any]]:
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

            include_in_trending, local_category = _classify_item(title, description, section_name)

            data.append(
                {
                    "model_id": title,
                    "url": _clean_url(article_url),
                    "description": description,
                    "tags": [section_name],
                    "source": "tldr_ai",
                    "include_in_trending": include_in_trending,
                    "tldr_local_category": local_category,
                    "edition_date": edition_date or date.today().isoformat(),
                }
            )

    return data


# Register a default instance in the global registry.
registry.register(TLDRAIWatcher())
