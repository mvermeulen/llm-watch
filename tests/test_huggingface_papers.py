"""Unit tests for Hugging Face trending papers watcher."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import requests

from llmwatch.agents.watchers.huggingface_papers import HuggingFaceTrendingPapersWatcher


def _iso(days_ago: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat().replace("+00:00", "Z")


def _mock_get_json(data, status_code=200):
    """Return a mock requests.Response that yields JSON data."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = data
    mock_resp.raise_for_status.return_value = None
    return mock_resp


class TestHuggingFaceTrendingPapersWatcher:
    def test_successful_fetch(self):
        payload = [
            {
                "title": "Paper A",
                "summary": "Summary A",
                "publishedAt": _iso(1),
                "paper": {
                    "id": "2604.00001",
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                    "upvotes": 12,
                },
            },
            {
                "title": "Paper B",
                "summary": "Summary B",
                "publishedAt": _iso(2),
                "paper": {
                    "id": "2604.00002",
                    "authors": [{"name": "Carol"}],
                    "upvotes": 3,
                },
            },
        ]
        with patch(
            "llmwatch.agents.watchers.huggingface_papers.requests.get",
            return_value=_mock_get_json(payload),
        ):
            agent = HuggingFaceTrendingPapersWatcher()
            result = agent.run(context={"hf_papers_limit": 5, "hf_papers_lookback_days": 7})

        assert result.ok()
        assert result.agent_name == "huggingface_trending_papers"
        assert result.category == "watcher"
        assert len(result.data) == 2
        assert result.data[0]["source"] == "huggingface_papers"
        assert result.data[0]["url"].startswith("https://huggingface.co/papers/")

    def test_filters_out_old_papers(self):
        payload = [
            {
                "title": "Old Paper",
                "summary": "Old summary",
                "publishedAt": _iso(15),
                "paper": {"id": "2603.00001", "authors": [], "upvotes": 1},
            }
        ]
        with patch(
            "llmwatch.agents.watchers.huggingface_papers.requests.get",
            return_value=_mock_get_json(payload),
        ):
            agent = HuggingFaceTrendingPapersWatcher()
            result = agent.run(context={"hf_papers_lookback_days": 7})

        assert result.ok()
        assert result.data == []

    def test_request_error_returns_error_result(self):
        with patch(
            "llmwatch.agents.watchers.huggingface_papers.requests.get",
            side_effect=requests.ConnectionError("timeout"),
        ):
            agent = HuggingFaceTrendingPapersWatcher()
            result = agent.run()

        assert not result.ok()
        assert result.data == []
        assert any("timeout" in e for e in result.errors)
