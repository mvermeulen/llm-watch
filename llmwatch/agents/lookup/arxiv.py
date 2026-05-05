"""
arXiv paper lookup agent.

Given model names / topics collected by the watcher agents, this agent
queries the arXiv Atom API and returns matching recent papers.

arXiv API documentation: https://info.arxiv.org/help/api/index.html
"""

from __future__ import annotations

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote_plus

import requests

from llmwatch.agents.base import AgentResult, BaseAgent, registry

logger = logging.getLogger(__name__)

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_MAX_RESULTS_PER_QUERY = 3
_REQUEST_TIMEOUT = 20
from llmwatch.cache import get_cache_dir as _get_cache_dir
_CACHE_PATH = os.path.join(_get_cache_dir(), "arxiv_lookup_cache.json")

# arXiv Atom namespace
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


class ArxivLookupAgent(BaseAgent):
    """
    Search arXiv for papers related to LLMs discovered by watcher agents.

    The agent reads ``context["watcher_results"]`` (a list of
    :class:`~llmwatch.agents.base.AgentResult` objects) to build a list of
    search terms from the model names and tags found by the watchers, then
    fires off arXiv API queries.

    Each item in ``AgentResult.data`` is a dict with:

    * ``title``    – paper title
    * ``authors``  – comma-separated author list
    * ``summary``  – abstract (truncated to 300 chars)
    * ``url``      – arXiv abstract URL
    * ``published``– ISO-8601 publication date
    * ``query``    – the search term that found this paper
    """

    name = "arxiv_lookup"
    category = "lookup"

    # Maximum distinct search terms to query (avoids hammering the API)
    max_terms: int = 10

    def run(self, context: dict[str, Any] | None = None) -> AgentResult:
        cache_first = _cache_first_enabled(context)
        terms = _extract_search_terms(context)
        if not terms:
            logger.info("ArxivLookupAgent: no search terms from context – using default query")
            terms = ["large language model"]

        terms = terms[: self.max_terms]
        logger.info(
            "ArxivLookupAgent: resolving %d term(s) with cache_first=%s: %s",
            len(terms),
            cache_first,
            terms,
        )

        all_data: list[dict[str, Any]] = []
        errors: list[str] = []
        cache = _load_cache()

        for term in terms:
            cache_key = _cache_key(term)
            if cache_first and cache_key in cache:
                logger.info("ArxivLookupAgent: cache hit for term '%s'", term)
                all_data.extend(cache[cache_key])
                continue

            try:
                papers = _query_arxiv(term)
                all_data.extend(papers)
                cache[cache_key] = papers
            except requests.RequestException as exc:
                msg = f"arXiv query failed for '{term}': {exc}"
                logger.warning(msg)
                errors.append(msg)

        _save_cache(cache)

        # De-duplicate by URL
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for paper in all_data:
            if paper["url"] not in seen:
                seen.add(paper["url"])
                unique.append(paper)

        logger.info("ArxivLookupAgent: found %d unique papers", len(unique))
        return self._result(data=unique, errors=errors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_search_terms(context: dict[str, Any] | None) -> list[str]:
    """Derive search terms from watcher results stored in *context*."""
    if not context:
        return []

    terms: list[str] = []
    watcher_results = context.get("watcher_results", [])

    for result in watcher_results:
        for item in result.data:
            # Use the model_id as a candidate term (strip org prefix)
            model_id: str = item.get("model_id", "")
            if model_id:
                # "mistralai/Mistral-7B" → "Mistral-7B"
                short = model_id.split("/")[-1] if "/" in model_id else model_id
                # Convert hyphens/underscores to spaces and clean up version tags
                short = re.sub(r"[-_]", " ", short)
                short = re.sub(r"\b(v?\d+[\.\d]*)\b", "", short).strip()
                if len(short) > 3 and short not in terms:
                    terms.append(short)

    return terms


def _cache_first_enabled(context: dict[str, Any] | None) -> bool:
    """Return True unless lookup options explicitly request force-fetch."""
    if not context:
        return True
    options = context.get("options", {})
    if not isinstance(options, dict):
        return True
    return not bool(options.get("arxiv_force_fetch", False))


def _cache_key(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def _load_cache() -> dict[str, list[dict[str, Any]]]:
    """Load cached arXiv results from disk; return an empty cache on errors."""
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            cache = json.load(fh)
        if isinstance(cache, dict):
            return cache
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("ArxivLookupAgent: failed to load cache: %s", exc)
    return {}


def _save_cache(cache: dict[str, list[dict[str, Any]]]) -> None:
    """Persist cache to disk. Errors are non-fatal for report generation."""
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("ArxivLookupAgent: failed to save cache: %s", exc)


def _query_arxiv(query: str, max_results: int = _MAX_RESULTS_PER_QUERY) -> list[dict[str, Any]]:
    """Fire a single arXiv API query and parse the Atom feed response."""
    params = {
        "search_query": f"all:{quote_plus(query)}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(_ARXIV_API_URL, params=params, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    return _parse_atom_feed(resp.text, query)


def _parse_atom_feed(xml_text: str, query: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    papers: list[dict[str, Any]] = []

    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        title_el = entry.find(f"{{{_ATOM_NS}}}title")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        summary_el = entry.find(f"{{{_ATOM_NS}}}summary")
        summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
        summary = summary[:300] + ("…" if len(summary) > 300 else "")

        published_el = entry.find(f"{{{_ATOM_NS}}}published")
        published = published_el.text.strip() if published_el is not None and published_el.text else ""

        url = ""
        for link in entry.findall(f"{{{_ATOM_NS}}}link"):
            if link.attrib.get("rel") == "alternate":
                url = link.attrib.get("href", "")
                break
        if not url:
            id_el = entry.find(f"{{{_ATOM_NS}}}id")
            url = id_el.text.strip() if id_el is not None and id_el.text else ""

        authors = []
        for author_el in entry.findall(f"{{{_ATOM_NS}}}author"):
            name_el = author_el.find(f"{{{_ATOM_NS}}}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        papers.append(
            {
                "title": title,
                "authors": ", ".join(authors),
                "summary": summary,
                "url": url,
                "published": published[:10],  # keep only date part
                "query": query,
            }
        )

    return papers


# Register a default instance in the global registry.
registry.register(ArxivLookupAgent())
