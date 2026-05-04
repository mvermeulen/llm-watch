"""
Vendor blog watchers using lightweight HTML scraping (Phase 2).

This module targets sources without stable RSS/Atom feeds and extracts post
links from listing pages.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urljoin

import requests

from llmwatch.agents.base import BaseAgent, registry

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 20
_DEFAULT_LOOKBACK_DAYS = 14
_DEFAULT_MAX_ITEMS = 20
_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_HEALTH_WARNING_STREAK = 2
_CACHE_DIR = ".llmwatch_cache"
_HEALTH_CACHE_PATH = os.path.join(_CACHE_DIR, "vendor_scrape_health.json")
_HEALTH_LOCK = threading.Lock()

_A_TAG_RE = re.compile(
    r'(<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>)(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TITLE_ATTR_RE = re.compile(r'(?:title|aria-label)=["\']([^"\']+)["\']', re.IGNORECASE)

_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)

_CHALLENGE_MARKERS = (
    "just a moment",
    "attention required",
    "verify you are human",
    "captcha",
    "cf-chl",
)

_GENERIC_LOW_SIGNAL_TITLES = {
    "about",
    "blog",
    "careers",
    "company",
    "contact",
    "featured",
    "home",
    "learn more",
    "more",
    "news",
    "products",
    "read",
}

_MIN_DESCRIPTION_LEN = 24


@dataclass(frozen=True)
class VendorScrapeConfig:
    """Configuration for a vendor listing page scrape watcher."""

    agent_name: str
    label: str
    source_slug: str
    listing_url: str
    post_url_re: re.Pattern[str]


_VENDOR_SCRAPE_CONFIGS: tuple[VendorScrapeConfig, ...] = (
    VendorScrapeConfig(
        agent_name="meta_ai_blog_scrape",
        label="Meta AI Blog",
        source_slug="meta_ai_blog",
        listing_url="https://ai.meta.com/blog/",
        post_url_re=re.compile(r"^https://ai\.meta\.com/blog/[a-z0-9][a-z0-9\-]*/?$", re.IGNORECASE),
    ),
    VendorScrapeConfig(
        agent_name="anthropic_news_scrape",
        label="Anthropic News",
        source_slug="anthropic_news",
        listing_url="https://www.anthropic.com/news",
        post_url_re=re.compile(r"^https://www\.anthropic\.com/news/[a-z0-9][a-z0-9\-]*/?$", re.IGNORECASE),
    ),
    VendorScrapeConfig(
        agent_name="mistral_news_scrape",
        label="Mistral News",
        source_slug="mistral_news",
        listing_url="https://mistral.ai/news",
        post_url_re=re.compile(r"^https://mistral\.ai/news/[a-z0-9][a-z0-9\-]*/?$", re.IGNORECASE),
    ),
    VendorScrapeConfig(
        agent_name="xai_news_scrape",
        label="xAI News",
        source_slug="xai_news",
        listing_url="https://x.ai/news",
        post_url_re=re.compile(r"^https://x\.ai/news/[a-z0-9][a-z0-9\-]*/?$", re.IGNORECASE),
    ),
)

PHASE2_VENDOR_SCRAPE_AGENT_NAMES: tuple[str, ...] = tuple(
    cfg.agent_name for cfg in _VENDOR_SCRAPE_CONFIGS
)

PHASE2_VENDOR_SCRAPE_ALIASES: dict[str, str] = {
    "meta": "meta_ai_blog_scrape",
    "anthropic": "anthropic_news_scrape",
    "mistral": "mistral_news_scrape",
    "xai": "xai_news_scrape",
}


class VendorScrapeWatcher(BaseAgent):
    """Scrape vendor listing pages and extract post links."""

    category = "watcher"

    def __init__(self, config: VendorScrapeConfig) -> None:
        self.config = config
        self.name = config.agent_name

    def run(self, context: dict[str, Any] | None = None):
        context = context or {}
        lookback_days = int(context.get("vendor_scrape_lookback_days", _DEFAULT_LOOKBACK_DAYS))
        max_items = int(context.get("vendor_scrape_max_items", _DEFAULT_MAX_ITEMS))
        per_source_limits = context.get("vendor_scrape_per_source_max_items", {})
        if isinstance(per_source_limits, dict):
            per_source_value = per_source_limits.get(self.name)
            if per_source_value is not None:
                try:
                    max_items = int(per_source_value)
                except (TypeError, ValueError):
                    logger.warning(
                        "%s: ignoring invalid per-source max '%s'",
                        self.config.label,
                        per_source_value,
                    )

        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))
        max_items = max(1, max_items)
        retry_attempts = max(1, int(context.get("vendor_scrape_retry_attempts", _DEFAULT_RETRY_ATTEMPTS)))
        health_warning_streak = _health_warning_streak(context)

        logger.info("%s: scraping %s", self.config.label, self.config.listing_url)
        try:
            resp = _fetch_listing_with_retries(
                self.config.listing_url,
                attempts=retry_attempts,
            )
        except requests.RequestException as exc:
            _update_health_streak(self.name, outcome="error")
            msg = f"{self.config.label} request failed: {exc}"
            logger.error(msg)
            return self._result(errors=[msg])

        candidate_by_url: dict[str, tuple[int, int, dict[str, Any]]] = {}
        html_text = resp.text

        for idx, match in enumerate(_A_TAG_RE.finditer(html_text)):

            open_tag = match.group(1) or ""
            href_raw = html_lib.unescape((match.group(2) or "").strip())
            if not href_raw:
                continue
            href = urljoin(self.config.listing_url, href_raw)
            if not self.config.post_url_re.match(href):
                continue

            title = _clean_text(match.group(3) or "")
            if not title:
                title = _extract_title_fallback(open_tag)
            if not title:
                continue

            title = _normalize_title(
                title=title,
                url=href,
                source_slug=self.config.source_slug,
            )
            if not title:
                title = _derive_title_from_url(href)
                if not title:
                    continue

            if _is_low_signal_title(title):
                continue

            published_dt = _extract_nearby_date(html_text, match.start(), match.end())
            if published_dt is not None and published_dt < cutoff:
                continue

            description = _extract_nearby_description(
                html_text,
                start=match.start(),
                end=match.end(),
                title=title,
            )

            item = {
                "model_id": title,
                "url": href,
                "description": description,
                "tags": [self.config.source_slug, "scrape"],
                "source": self.config.source_slug,
                "published": (
                    published_dt.date().isoformat() if published_dt is not None else ""
                ),
            }
            score = _title_quality_score(title=title, description=description)
            existing = candidate_by_url.get(href)
            if existing is None:
                candidate_by_url[href] = (score, idx, item)
                continue

            existing_score, existing_idx, existing_item = existing
            existing_title = str(existing_item.get("model_id", ""))

            candidate_effective_score = score + _duplicate_specificity_bonus(
                candidate_title=title,
                existing_title=existing_title,
            )
            existing_effective_score = existing_score + _duplicate_specificity_bonus(
                candidate_title=existing_title,
                existing_title=title,
            )

            if candidate_effective_score > existing_effective_score or (
                candidate_effective_score == existing_effective_score and idx < existing_idx
            ):
                candidate_by_url[href] = (score, idx, item)

        ranked = sorted(candidate_by_url.values(), key=lambda x: x[1])
        data = [entry[2] for entry in ranked[:max_items]]

        previous_streak, current_streak = _update_health_streak(
            self.name,
            outcome="zero" if not data else "ok",
        )

        if not data and current_streak >= health_warning_streak:
            logger.warning(
                "%s: scrape health warning - 0 items collected from %s; "
                "consecutive zero-item streak=%d (threshold=%d)",
                self.config.label,
                self.config.listing_url,
                current_streak,
                health_warning_streak,
            )
        elif data and previous_streak >= health_warning_streak:
            logger.info(
                "%s: scrape health recovered after zero-item streak of %d",
                self.config.label,
                previous_streak,
            )

        logger.info("%s: collected %d scraped item(s)", self.config.label, len(data))
        return self._result(data=data)


def _clean_text(text: str) -> str:
    plain = _STRIP_TAGS_RE.sub("", html_lib.unescape(text or "")).strip()
    return _WS_RE.sub(" ", plain)


def _extract_title_fallback(open_tag: str) -> str:
    match = _TITLE_ATTR_RE.search(open_tag or "")
    if not match:
        return ""
    return _clean_text(match.group(1))


def _is_low_signal_title(title: str) -> bool:
    return title.strip().lower() in _GENERIC_LOW_SIGNAL_TITLES


def _normalize_title(title: str, url: str, source_slug: str) -> str:
    normalized = _clean_text(title)
    if source_slug == "anthropic_news":
        normalized = _cleanup_anthropic_title(normalized, url)
    return _clean_text(normalized)


def _cleanup_anthropic_title(title: str, url: str) -> str:
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)
    cleaned = re.sub(r"([0-9])([A-Z])", r"\1 \2", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()

    # Drop embedded section/date metadata often concatenated in Anthropic cards.
    section = r"(?:Product|Announcement|Announcements|Research|Policy|Safety|Company)"
    month = (
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)"
    )
    date_text = rf"{month}\s+\d{{1,2}},\s+\d{{4}}"

    cleaned = re.sub(rf"\b{section}\s*{date_text}\b", " ", cleaned)
    cleaned = re.sub(rf"\b{date_text}\s*{section}\b", " ", cleaned)
    cleaned = re.sub(rf"^{date_text}\s+", "", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip(" -:\u00a0")

    noisy = (
        len(cleaned) > 100
        or "today," in cleaned.lower()
        or bool(re.search(r"\b(?:our|we|this)\b", cleaned.lower())) and len(cleaned) > 70
    )
    if noisy:
        derived = _derive_title_from_url(url)
        if derived:
            return derived

    return cleaned


def _derive_title_from_url(url: str) -> str:
    path = (url or "").split("?", 1)[0].rstrip("/")
    slug = path.rsplit("/", 1)[-1].strip()
    if not slug:
        return ""

    words = [part for part in re.split(r"[-_]+", slug) if part]
    if not words:
        return ""

    normalized_words: list[str] = []
    for word in words:
        if word.isdigit():
            normalized_words.append(word)
        elif len(word) <= 3 and word.isalpha() and word.lower() in {"ai", "api", "sdk", "llm"}:
            normalized_words.append(word.upper())
        else:
            normalized_words.append(word.capitalize())
    return " ".join(normalized_words)


def _extract_nearby_description(html_text: str, start: int, end: int, title: str) -> str:
    snippet = _clean_text(html_text[end:min(len(html_text), end + 280)])
    if not snippet:
        return ""

    # Remove duplicated leading title and trailing boilerplate-like fragments.
    lowered = snippet.lower()
    title_lower = title.lower()
    if lowered.startswith(title_lower):
        snippet = snippet[len(title):].lstrip(" -:\u00a0")

    if len(snippet) < _MIN_DESCRIPTION_LEN:
        return ""
    return snippet[:220]


def _title_quality_score(title: str, description: str) -> int:
    cleaned = _clean_text(title)
    if not cleaned:
        return -100

    score = 0
    if not _is_low_signal_title(cleaned):
        score += 5
    score += min(len(cleaned), 90) // 10
    if ":" in cleaned or "-" in cleaned:
        score += 1
    if description and len(description) >= _MIN_DESCRIPTION_LEN:
        score += 1
    return score


def _duplicate_specificity_bonus(candidate_title: str, existing_title: str) -> int:
    candidate = _clean_text(candidate_title).lower()
    existing = _clean_text(existing_title).lower()
    if not candidate or not existing or candidate == existing:
        return 0

    # Tiny tie-break for duplicates: if one title is a strict extension of another,
    # prefer the longer one as the likely more specific headline.
    if candidate.startswith(existing) and len(candidate) >= len(existing) + 8:
        return 1
    return 0


def _extract_nearby_date(html_text: str, start: int, end: int) -> datetime | None:
    # Prefer the closest preceding date to avoid using dates from neighboring cards.
    before_start = max(0, start - 260)
    before_snippet = _clean_text(html_text[before_start:start])
    before_matches = list(_DATE_RE.finditer(before_snippet))
    if before_matches:
        date_text = before_matches[-1].group(0)
    else:
        after_end = min(len(html_text), end + 200)
        after_snippet = _clean_text(html_text[end:after_end])
        after_match = _DATE_RE.search(after_snippet)
        if not after_match:
            return None
        date_text = after_match.group(0)

    patterns = ["%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"]
    for pattern in patterns:
        try:
            parsed = datetime.strptime(date_text, pattern)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _looks_like_challenge_page(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _CHALLENGE_MARKERS)


def _fetch_listing_with_retries(url: str, attempts: int) -> requests.Response:
    headers = {"User-Agent": "llm-watch/0.1 (https://github.com/mvermeulen/llm-watch)"}
    last_exc: Exception | None = None

    for attempt in range(1, max(1, attempts) + 1):
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers=headers)
            resp.raise_for_status()

            # Retry challenge pages since some providers intermittently serve them.
            if _looks_like_challenge_page(resp.text):
                raise requests.RequestException("challenge page detected")
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(0.35 * attempt)

    raise requests.RequestException(f"failed after {attempts} attempt(s): {last_exc}")


def _health_warning_streak(context: dict[str, Any] | None = None) -> int:
    context = context or {}
    value = context.get("vendor_scrape_health_warning_streak")
    if value is None:
        value = os.getenv(
            "LLMWATCH_VENDOR_SCRAPE_HEALTH_WARNING_STREAK",
            str(_DEFAULT_HEALTH_WARNING_STREAK),
        )
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return _DEFAULT_HEALTH_WARNING_STREAK


def _load_health_cache() -> dict[str, Any]:
    if not os.path.exists(_HEALTH_CACHE_PATH):
        return {"streaks": {}}
    try:
        with open(_HEALTH_CACHE_PATH, encoding="utf-8") as fh:
            parsed = json.load(fh)
        if not isinstance(parsed, dict):
            return {"streaks": {}}
        if "streaks" not in parsed or not isinstance(parsed["streaks"], dict):
            parsed["streaks"] = {}
        return parsed
    except (OSError, json.JSONDecodeError):
        return {"streaks": {}}


def _save_health_cache(cache: dict[str, Any]) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_HEALTH_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)


def _update_health_streak(agent_name: str, outcome: str) -> tuple[int, int]:
    with _HEALTH_LOCK:
        cache = _load_health_cache()
        streaks = cache.setdefault("streaks", {})
        current = streaks.get(agent_name, {})
        previous = int(current.get("count", 0)) if isinstance(current, dict) else 0

        if outcome == "zero":
            new_value = previous + 1
        else:
            new_value = 0

        streaks[agent_name] = {
            "count": new_value,
            "last_outcome": outcome,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_health_cache(cache)
        return previous, new_value


def get_health_streak(agent_name: str) -> int:
    with _HEALTH_LOCK:
        cache = _load_health_cache()
        value = cache.get("streaks", {}).get(agent_name, {})
        if not isinstance(value, dict):
            return 0
        try:
            return int(value.get("count", 0))
        except (TypeError, ValueError):
            return 0


def get_health_warning_threshold(context: dict[str, Any] | None = None) -> int:
    return _health_warning_streak(context)


for _cfg in _VENDOR_SCRAPE_CONFIGS:
    registry.register(VendorScrapeWatcher(_cfg))
