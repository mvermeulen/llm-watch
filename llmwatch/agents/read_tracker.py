"""
Read tracker – persist URLs the user has already read so they are suppressed
from future reports.

Storage
-------
Entries are written to ``$LLMWATCH_CACHE_DIR/read_urls.json`` when
``LLMWATCH_CACHE_DIR`` is set, otherwise ``.llmwatch_cache/read_urls.json``.
Each entry stores the normalized URL (query-string / fragment stripped), an
optional title, and the ISO date it was marked read.

CLI usage (via ``llm-watch``)::

    llm-watch --mark-read https://example.com/article
    llm-watch --mark-read-from-report llm_watch_report_2026-05-05.md
    llm-watch --mark-read-from-report report.md --section "Common Links"
    llm-watch --list-read
    llm-watch --clear-read
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from urllib.parse import urlparse

from llmwatch.cache import get_cache_dir

# Matches Markdown inline links: [Title](URL) — captures title and URL separately.
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^)\s]+)\)')

# Matches bare H2 section headings: ## Section Name
_H2_RE = re.compile(r'^##\s+(.+)', re.MULTILINE)

logger = logging.getLogger(__name__)


def _read_urls_path() -> str:
    return os.path.join(get_cache_dir(), "read_urls.json")


# Stable public constant kept for backwards compatibility and test patching.
READ_URLS_PATH = _read_urls_path()

_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# URL helpers (mirrors consolidator logic to ensure consistent normalization)
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Strip query parameters, fragments, and trailing slashes for stable comparison."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _load_raw() -> dict:
    """Load the raw JSON store.  Returns an empty schema-valid dict on any error."""
    path = _read_urls_path()
    if not os.path.exists(path):
        return {"version": _SCHEMA_VERSION, "entries": {}}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "entries" not in data:
            logger.warning("read_tracker: unexpected format in %s, resetting", path)
            return {"version": _SCHEMA_VERSION, "entries": {}}
        return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("read_tracker: could not read %s: %s", path, exc)
        return {"version": _SCHEMA_VERSION, "entries": {}}


def _save_raw(data: dict) -> None:
    path = _read_urls_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def _extract_section(text: str, section: str) -> str:
    """
    Return the portion of *text* that falls under the H2 heading whose title
    contains *section* (case-insensitive).  Returns the full text if no
    matching heading is found.
    """
    matches = list(_H2_RE.finditer(text))
    for i, m in enumerate(matches):
        if section.lower() in m.group(1).lower():
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            return text[start:end]
    return ""


def parse_report_urls(
    report_path: str,
    section: str | None = None,
) -> dict[str, str]:
    """
    Extract all Markdown hyperlinks from a generated report file.

    Parameters
    ----------
    report_path:
        Path to the ``.md`` report file.
    section:
        Optional H2 section name filter (e.g. ``"Common Links"``).  When
        supplied only URLs found within that section are returned.  The match
        is case-insensitive and partial (substring).

    Returns
    -------
    dict[str, str]
        Mapping of raw URL → link title for every ``[title](url)`` found.
        Duplicate URLs keep the title from their first occurrence.
    """
    try:
        with open(report_path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read report file '{report_path}': {exc}") from exc

    if section:
        scoped = _extract_section(text, section)
        if not scoped:
            logger.warning(
                "parse_report_urls: section '%s' not found in '%s'",
                section,
                report_path,
            )
            return {}
        text = scoped

    result: dict[str, str] = {}
    for m in _MD_LINK_RE.finditer(text):
        title, url = m.group(1), m.group(2)
        if url not in result:
            result[url] = title
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_read_urls() -> frozenset[str]:
    """Return a frozenset of normalized URLs that have been marked as read."""
    data = _load_raw()
    return frozenset(data.get("entries", {}).keys())


def mark_read(urls: list[str], titles: dict[str, str] | None = None) -> int:
    """
    Mark one or more URLs as read.

    Parameters
    ----------
    urls:
        Raw URLs to mark (will be normalized before storing).
    titles:
        Optional mapping of raw URL → display title for richer metadata.

    Returns
    -------
    int
        Number of *new* entries added (already-present URLs are skipped).
    """
    titles = titles or {}
    data = _load_raw()
    entries: dict[str, dict] = data.setdefault("entries", {})
    today = date.today().isoformat()
    added = 0

    for url in urls:
        norm = normalize_url(url)
        if not norm:
            logger.warning("read_tracker: could not normalize URL '%s', skipping", url)
            continue
        if norm not in entries:
            entry: dict = {"marked_at": today}
            title = titles.get(url) or titles.get(norm)
            if title:
                entry["title"] = title
            entries[norm] = entry
            added += 1
            logger.debug("read_tracker: marked as read: %s", norm)
        else:
            logger.debug("read_tracker: already marked: %s", norm)

    if added:
        _save_raw(data)

    return added


def list_read() -> list[dict]:
    """
    Return all read entries as a list of dicts sorted by ``marked_at`` descending.

    Each dict has keys: ``url``, ``marked_at``, and optionally ``title``.
    """
    data = _load_raw()
    result = []
    for norm_url, meta in data.get("entries", {}).items():
        entry = {"url": norm_url, "marked_at": meta.get("marked_at", "")}
        if "title" in meta:
            entry["title"] = meta["title"]
        result.append(entry)
    return sorted(result, key=lambda e: e["marked_at"], reverse=True)


def clear_read() -> int:
    """
    Remove all read URL entries.

    Returns
    -------
    int
        Number of entries removed.
    """
    data = _load_raw()
    count = len(data.get("entries", {}))
    data["entries"] = {}
    _save_raw(data)
    return count


def unmark_read(urls: list[str]) -> int:
    """
    Remove specific URLs from the read list.

    Returns
    -------
    int
        Number of entries removed.
    """
    data = _load_raw()
    entries = data.get("entries", {})
    removed = 0
    for url in urls:
        norm = normalize_url(url)
        if norm and norm in entries:
            del entries[norm]
            removed += 1
    if removed:
        _save_raw(data)
    return removed
