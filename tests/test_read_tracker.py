"""
Tests for llmwatch.agents.read_tracker.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from llmwatch.agents.read_tracker import (
    READ_URLS_PATH,
    clear_read,
    list_read,
    load_read_urls,
    mark_read,
    normalize_url,
    parse_report_urls,
    unmark_read,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_path(tmp_path):
    """Redirect the cache directory to a temp folder via the env var."""
    return patch.dict("os.environ", {"LLMWATCH_CACHE_DIR": str(tmp_path)})


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------

class TestNormalizeUrl:
    def test_strips_query_string(self):
        assert normalize_url("https://example.com/article?utm_source=tldr") == "https://example.com/article"

    def test_strips_fragment(self):
        assert normalize_url("https://example.com/article#section") == "https://example.com/article"

    def test_strips_trailing_slash(self):
        assert normalize_url("https://example.com/article/") == "https://example.com/article"

    def test_empty_string(self):
        assert normalize_url("") == ""

    def test_preserves_path(self):
        assert normalize_url("https://arxiv.org/abs/2401.00001") == "https://arxiv.org/abs/2401.00001"

    def test_strips_both_query_and_fragment(self):
        url = "https://example.com/post?ref=hn#comments"
        assert normalize_url(url) == "https://example.com/post"


# ---------------------------------------------------------------------------
# mark_read / load_read_urls
# ---------------------------------------------------------------------------

class TestMarkRead:
    def test_mark_single_url(self, tmp_path):
        with _patch_path(tmp_path):
            added = mark_read(["https://example.com/article"])
            assert added == 1
            read = load_read_urls()
            assert "https://example.com/article" in read

    def test_mark_normalizes_url(self, tmp_path):
        with _patch_path(tmp_path):
            mark_read(["https://example.com/article?utm_source=tldr"])
            read = load_read_urls()
            assert "https://example.com/article" in read
            assert "https://example.com/article?utm_source=tldr" not in read

    def test_mark_multiple_urls(self, tmp_path):
        with _patch_path(tmp_path):
            added = mark_read(["https://a.com/1", "https://b.com/2"])
            assert added == 2
            read = load_read_urls()
            assert len(read) == 2

    def test_duplicate_not_counted(self, tmp_path):
        with _patch_path(tmp_path):
            mark_read(["https://example.com/article"])
            added = mark_read(["https://example.com/article"])
            assert added == 0
            assert len(load_read_urls()) == 1

    def test_skips_empty_url(self, tmp_path):
        with _patch_path(tmp_path):
            added = mark_read([""])
            assert added == 0
            assert len(load_read_urls()) == 0

    def test_stores_title_when_provided(self, tmp_path):
        with _patch_path(tmp_path):
            url = "https://example.com/article"
            mark_read([url], titles={url: "My Article"})
            entries = list_read()
            assert entries[0]["title"] == "My Article"

    def test_stores_marked_at_date(self, tmp_path):
        with _patch_path(tmp_path):
            mark_read(["https://example.com/x"])
            entries = list_read()
            assert len(entries[0]["marked_at"]) == 10  # YYYY-MM-DD


# ---------------------------------------------------------------------------
# load_read_urls
# ---------------------------------------------------------------------------

class TestLoadReadUrls:
    def test_empty_when_no_file(self, tmp_path):
        with _patch_path(tmp_path):
            assert load_read_urls() == frozenset()

    def test_returns_frozenset(self, tmp_path):
        with _patch_path(tmp_path):
            mark_read(["https://example.com/a"])
            result = load_read_urls()
            assert isinstance(result, frozenset)

    def test_handles_corrupt_file(self, tmp_path):
        with _patch_path(tmp_path):
            path = str(tmp_path / "read_urls.json")
            with open(path, "w") as f:
                f.write("NOT JSON{{")
            assert load_read_urls() == frozenset()

    def test_handles_unexpected_schema(self, tmp_path):
        with _patch_path(tmp_path):
            path = str(tmp_path / "read_urls.json")
            with open(path, "w") as f:
                json.dump(["unexpected", "list"], f)
            assert load_read_urls() == frozenset()


# ---------------------------------------------------------------------------
# list_read
# ---------------------------------------------------------------------------

class TestListRead:
    def test_empty_list(self, tmp_path):
        with _patch_path(tmp_path):
            assert list_read() == []

    def test_sorted_by_date_descending(self, tmp_path):
        with _patch_path(tmp_path):
            # Manually write two entries with different dates
            path = str(tmp_path / "read_urls.json")
            data = {
                "version": 1,
                "entries": {
                    "https://example.com/old": {"marked_at": "2026-04-01"},
                    "https://example.com/new": {"marked_at": "2026-05-01"},
                },
            }
            with open(path, "w") as f:
                json.dump(data, f)
            entries = list_read()
            assert entries[0]["url"] == "https://example.com/new"
            assert entries[1]["url"] == "https://example.com/old"


# ---------------------------------------------------------------------------
# unmark_read
# ---------------------------------------------------------------------------

class TestUnmarkRead:
    def test_removes_url(self, tmp_path):
        with _patch_path(tmp_path):
            mark_read(["https://example.com/a", "https://example.com/b"])
            removed = unmark_read(["https://example.com/a"])
            assert removed == 1
            read = load_read_urls()
            assert "https://example.com/a" not in read
            assert "https://example.com/b" in read

    def test_normalizes_before_removing(self, tmp_path):
        with _patch_path(tmp_path):
            mark_read(["https://example.com/a"])
            removed = unmark_read(["https://example.com/a?ref=x"])
            assert removed == 1
            assert load_read_urls() == frozenset()

    def test_missing_url_not_counted(self, tmp_path):
        with _patch_path(tmp_path):
            removed = unmark_read(["https://nothere.com/x"])
            assert removed == 0


# ---------------------------------------------------------------------------
# clear_read
# ---------------------------------------------------------------------------

class TestClearRead:
    def test_clears_all_entries(self, tmp_path):
        with _patch_path(tmp_path):
            mark_read(["https://a.com/1", "https://b.com/2"])
            removed = clear_read()
            assert removed == 2
            assert load_read_urls() == frozenset()

    def test_clear_empty_list(self, tmp_path):
        with _patch_path(tmp_path):
            removed = clear_read()
            assert removed == 0


# ---------------------------------------------------------------------------
# Integration: consolidator suppresses read URLs
# ---------------------------------------------------------------------------

class TestConsolidatorSuppressesReadUrls:
    def test_read_url_suppressed(self, tmp_path):
        from llmwatch.agents.consolidator import StoryConsolidatorAgent
        from llmwatch.agents.base import AgentResult

        url = "https://example.com/important-story"

        with _patch_path(tmp_path):
            mark_read([url])
            agent = StoryConsolidatorAgent()

            watcher_results = [
                AgentResult(
                    agent_name="tldr_ai",
                    category="watcher",
                    data=[{"name": "Important Story", "url": url, "description": "desc"}],
                )
            ]
            result = agent.run(context={"watcher_results": watcher_results, "lookup_results": []})

        suppressed = [s for s in result.data if s.get("suppression_reason") == "already_read"]
        assert len(suppressed) == 1

    def test_unread_url_not_suppressed(self, tmp_path):
        from llmwatch.agents.consolidator import StoryConsolidatorAgent
        from llmwatch.agents.base import AgentResult

        url = "https://example.com/unread-story"

        with _patch_path(tmp_path):
            agent = StoryConsolidatorAgent()
            watcher_results = [
                AgentResult(
                    agent_name="tldr_ai",
                    category="watcher",
                    data=[{"name": "Unread Story", "url": url, "description": "desc"}],
                )
            ]
            result = agent.run(context={"watcher_results": watcher_results, "lookup_results": []})

        read_suppressed = [s for s in result.data if s.get("suppression_reason") == "already_read"]
        assert len(read_suppressed) == 0

    def test_mark_read_after_construction_is_visible(self, tmp_path):
        """run() must reload read URLs on each call, not snapshot them at __init__."""
        from llmwatch.agents.consolidator import StoryConsolidatorAgent
        from llmwatch.agents.base import AgentResult

        url = "https://example.com/late-marked"

        with _patch_path(tmp_path):
            # Construct the agent BEFORE marking the URL.
            agent = StoryConsolidatorAgent()
            mark_read([url])  # mark happens after construction

            watcher_results = [
                AgentResult(
                    agent_name="tldr_ai",
                    category="watcher",
                    data=[{"name": "Late Marked Story", "url": url, "description": "desc"}],
                )
            ]
            result = agent.run(context={"watcher_results": watcher_results, "lookup_results": []})

        suppressed = [s for s in result.data if s.get("suppression_reason") == "already_read"]
        assert len(suppressed) == 1, "URL marked after agent construction should still be suppressed"


# ---------------------------------------------------------------------------
# parse_report_urls
# ---------------------------------------------------------------------------

_SAMPLE_REPORT = """\
# LLM Watch – Weekly Investigation Report
*Generated: 2026-05-05*

## Common Links This Week

### 1. [Great AI Story](https://example.com/story-one)

Some description here.

### 2. [Another Story](https://example.com/story-two?utm_source=tldr)

More text.

---

## Trending & New Models

- [Cool Model](https://huggingface.co/org/model)
- [Paper Title](https://arxiv.org/abs/2401.00001)
"""


class TestParseReportUrls:
    def test_extracts_all_links(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text(_SAMPLE_REPORT)
        result = parse_report_urls(str(report))
        assert "https://example.com/story-one" in result
        assert "https://example.com/story-two?utm_source=tldr" in result
        assert "https://huggingface.co/org/model" in result
        assert "https://arxiv.org/abs/2401.00001" in result

    def test_captures_titles(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text(_SAMPLE_REPORT)
        result = parse_report_urls(str(report))
        assert result["https://example.com/story-one"] == "Great AI Story"

    def test_section_filter_restricts_results(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text(_SAMPLE_REPORT)
        result = parse_report_urls(str(report), section="Common Links")
        assert "https://example.com/story-one" in result
        assert "https://example.com/story-two?utm_source=tldr" in result
        # Trending section links should not appear
        assert "https://huggingface.co/org/model" not in result

    def test_section_filter_case_insensitive(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text(_SAMPLE_REPORT)
        result = parse_report_urls(str(report), section="common links")
        assert "https://example.com/story-one" in result

    def test_section_filter_partial_match(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text(_SAMPLE_REPORT)
        result = parse_report_urls(str(report), section="Trending")
        assert "https://huggingface.co/org/model" in result
        assert "https://example.com/story-one" not in result

    def test_unknown_section_returns_empty(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text(_SAMPLE_REPORT)
        result = parse_report_urls(str(report), section="Does Not Exist")
        assert result == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_report_urls(str(tmp_path / "nonexistent.md"))

    def test_no_duplicate_urls(self, tmp_path):
        text = "## Section\n\n[Title](https://example.com/x)\n[Same](https://example.com/x)\n"
        report = tmp_path / "report.md"
        report.write_text(text)
        result = parse_report_urls(str(report))
        assert len([k for k in result if k == "https://example.com/x"]) == 1

    def test_empty_report_returns_empty(self, tmp_path):
        report = tmp_path / "report.md"
        report.write_text("# Just a heading\nNo links here.\n")
        assert parse_report_urls(str(report)) == {}

    def test_mark_read_from_report_round_trip(self, tmp_path):
        """parse_report_urls + mark_read correctly stores titles and normalized URLs."""
        report = tmp_path / "report.md"
        report.write_text(_SAMPLE_REPORT)

        with _patch_path(tmp_path):
            url_titles = parse_report_urls(str(report))
            added = mark_read(list(url_titles.keys()), titles=url_titles)
            assert added == len(url_titles)
            read = load_read_urls()
            # URL with query string should be stored normalized
            assert "https://example.com/story-two" in read
            assert "https://example.com/story-one" in read
