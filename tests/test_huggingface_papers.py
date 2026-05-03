"""Unit tests for Hugging Face trending papers watcher."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests
import responses as resp_lib

from llmwatch.agents.watchers.huggingface_papers import HuggingFaceTrendingPapersWatcher


def _iso(days_ago: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat().replace("+00:00", "Z")


class TestHuggingFaceTrendingPapersWatcher:
    @resp_lib.activate
    def test_successful_fetch(self):
        resp_lib.add(
            resp_lib.GET,
            "https://huggingface.co/api/daily_papers",
            json=[
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
            ],
            status=200,
        )

        agent = HuggingFaceTrendingPapersWatcher()
        result = agent.run(context={"hf_papers_limit": 5, "hf_papers_lookback_days": 7})

        assert result.ok()
        assert result.agent_name == "huggingface_trending_papers"
        assert result.category == "watcher"
        assert len(result.data) == 2
        assert result.data[0]["source"] == "huggingface_papers"
        assert result.data[0]["url"].startswith("https://huggingface.co/papers/")

    @resp_lib.activate
    def test_filters_out_old_papers(self):
        resp_lib.add(
            resp_lib.GET,
            "https://huggingface.co/api/daily_papers",
            json=[
                {
                    "title": "Old Paper",
                    "summary": "Old summary",
                    "publishedAt": _iso(15),
                    "paper": {"id": "2603.00001", "authors": [], "upvotes": 1},
                }
            ],
            status=200,
        )

        agent = HuggingFaceTrendingPapersWatcher()
        result = agent.run(context={"hf_papers_lookback_days": 7})

        assert result.ok()
        assert result.data == []

    @resp_lib.activate
    def test_request_error_returns_error_result(self):
        resp_lib.add(
            resp_lib.GET,
            "https://huggingface.co/api/daily_papers",
            body=requests.ConnectionError("timeout"),
        )

        agent = HuggingFaceTrendingPapersWatcher()
        result = agent.run()

        assert not result.ok()
        assert result.data == []
        assert any("timeout" in e for e in result.errors)
