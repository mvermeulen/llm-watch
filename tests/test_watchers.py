"""
Unit tests for the watcher agents using mocked HTTP responses.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from llmwatch.agents.watchers.huggingface import HuggingFaceTrendingWatcher, _extract_urls_from_card
from llmwatch.agents.watchers.ollama import OllamaModelWatcher, _parse_ollama_library
from llmwatch.agents.watchers import tldr_ai as tldr_mod


def _mock_get_json(data, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = data
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = requests.HTTPError(str(status_code))
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


def _mock_get_text(body, status_code=200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = body
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---- HuggingFace watcher ------------------------------------------------- #

HF_SAMPLE = [
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "author": "mistralai",
        "downloads": 1_000_000,
        "likes": 5000,
        "tags": ["text-generation", "transformers"],
        "description": "A fast and efficient instruction-following model.",
        "cardData": {"homepage": "https://mistral.ai"},
    },
    {
        "id": "meta-llama/Llama-3-8B",
        "author": "meta-llama",
        "downloads": 500_000,
        "likes": 3000,
        "tags": ["text-generation"],
        "description": "Meta's Llama 3 8B base model.",
        "cardData": {},
    },
]


class TestHuggingFaceTrendingWatcher:
    def test_successful_fetch(self):
        with patch(
            "llmwatch.agents.watchers.huggingface.requests.get",
            return_value=_mock_get_json(HF_SAMPLE),
        ):
            agent = HuggingFaceTrendingWatcher(limit=2)
            result = agent.run()

        assert result.ok()
        assert result.agent_name == "huggingface_trending"
        assert result.category == "watcher"
        assert len(result.data) == 2

        first = result.data[0]
        assert first["model_id"] == "mistralai/Mistral-7B-Instruct-v0.2"
        assert first["author"] == "mistralai"
        assert first["source"] == "huggingface"
        assert first["url"].startswith("https://huggingface.co/")

    def test_network_error_returns_error_result(self):
        with patch(
            "llmwatch.agents.watchers.huggingface.requests.get",
            side_effect=requests.ConnectionError("connection refused"),
        ):
            agent = HuggingFaceTrendingWatcher()
            result = agent.run()

        assert not result.ok()
        assert result.data == []
        assert any("connection refused" in e for e in result.errors)

    def test_http_error_returns_error_result(self):
        with patch(
            "llmwatch.agents.watchers.huggingface.requests.get",
            return_value=_mock_get_json({"error": "Internal Server Error"}, status_code=500),
        ):
            agent = HuggingFaceTrendingWatcher()
            result = agent.run()

        assert not result.ok()

    def test_new_sources_extracted_from_card_data(self):
        with patch(
            "llmwatch.agents.watchers.huggingface.requests.get",
            return_value=_mock_get_json(HF_SAMPLE),
        ):
            agent = HuggingFaceTrendingWatcher(limit=2)
            result = agent.run()

        # mistral.ai should be detected as a new source
        assert any("mistral.ai" in s for s in result.new_sources)

    def test_extract_urls_from_card(self):
        card = {
            "homepage": "https://example.com/model",
            "license": "apache-2.0",
            "sources": ["https://another.example.org/paper"],
        }
        urls = _extract_urls_from_card(card)
        assert "https://example.com/model" in urls
        assert "https://another.example.org/paper" in urls

    def test_extract_urls_from_card_empty(self):
        assert _extract_urls_from_card({}) == []


# ---- Ollama watcher ------------------------------------------------------ #

OLLAMA_HTML = """
<!DOCTYPE html>
<html>
<body>
  <ul>
    <li>
            <a href="/library/llama3.2">
                <span x-test-search-response-title>llama3.2</span>
                <p>Meta's Llama 3.2 model collection</p>
                <span x-test-capability>tools</span>
                <span x-test-size>3B</span>
                <span x-test-size>11B</span>
      </a>
    </li>
    <li>
            <a href="/library/mistral">
                <span x-test-search-response-title>mistral</span>
                <p>The Mistral 7B model</p>
                <span x-test-size>7B</span>
      </a>
    </li>
    <li>
      <a href="/library">should be skipped</a>
    </li>
  </ul>
</body>
</html>
"""


class TestOllamaModelWatcher:
    def test_successful_scrape(self):
        with patch(
            "llmwatch.agents.watchers.ollama.requests.get",
            return_value=_mock_get_text(OLLAMA_HTML),
        ):
            agent = OllamaModelWatcher()
            result = agent.run()

        assert result.ok()
        assert result.agent_name == "ollama_models"
        assert result.category == "watcher"
        # At least some models should be found
        model_ids = [d["model_id"] for d in result.data]
        assert "llama3.2" in model_ids or "mistral" in model_ids

    def test_network_error_returns_error_result(self):
        with patch(
            "llmwatch.agents.watchers.ollama.requests.get",
            side_effect=requests.ConnectionError("timeout"),
        ):
            agent = OllamaModelWatcher()
            result = agent.run()

        assert not result.ok()
        assert result.data == []

    def test_parse_ollama_library_skips_nav_paths(self):
        html = '<a href="/library">Library</a><a href="/library/llama3.2"><span x-test-search-response-title>llama3.2</span></a>'
        data = _parse_ollama_library(html)
        model_ids = [d["model_id"] for d in data]
        assert "library" not in model_ids

    def test_parse_ollama_library_deduplicates(self):
        html = (
            '<a href="/library/llama3.2"><span x-test-search-response-title>llama3.2</span></a>'
            '<a href="/library/llama3.2"><span x-test-search-response-title>llama3.2</span></a>'
        )
        data = _parse_ollama_library(html)
        model_ids = [d["model_id"] for d in data]
        assert model_ids.count("llama3.2") <= 1

    def test_parse_ollama_library_sets_source(self):
        html = '<a href="/library/gemma3"><span x-test-search-response-title>Gemma 3</span></a>'
        data = _parse_ollama_library(html)
        for item in data:
            assert item["source"] == "ollama"
            assert item["url"].startswith("https://ollama.com/")


class TestTldrHybridFilter:
    def test_classify_with_ollama_parses_valid_json(self, monkeypatch):
        class FakeResp:
            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {
                    "response": json.dumps(
                        {
                            "include_in_trending": True,
                            "category": "trending_new_models",
                        }
                    )
                }

        monkeypatch.setattr(tldr_mod.requests, "post", lambda *a, **k: FakeResp())

        result = tldr_mod._classify_with_ollama(
            title="xAI launched Grok 4.3",
            description="Model release.",
            section="Headlines & Launches",
        )

        assert result == (True, "trending_new_models")

    def test_classify_item_falls_back_to_rules_when_ollama_fails(self, monkeypatch):
        monkeypatch.setattr(tldr_mod, "_is_ollama_filter_enabled", lambda: True)
        monkeypatch.setattr(tldr_mod, "_classify_with_ollama", lambda *a, **k: None)

        include, category = tldr_mod._classify_item(
            title="New model release",
            description="A new LLM checkpoint is available.",
            section="Headlines & Launches",
        )

        assert include is True
        assert category == "trending_new_models"

    def test_parse_tldr_newsletter_includes_local_category(self, monkeypatch):
        monkeypatch.setattr(
            tldr_mod,
            "_classify_item",
            lambda *a, **k: (True, "model_analysis"),
        )

        html = """
        <section>
          <header><h3>Deep Dives & Analysis</h3></header>
          <article>
            <a class="font-bold" href="https://example.com/kv-cache">
              <h3>KV Cache Locality (4 minute read)</h3>
            </a>
            <div class="newsletter-html">A detailed analysis.</div>
          </article>
        </section>
        """
        data = tldr_mod._parse_tldr_newsletter(html)

        assert len(data) == 1
        assert data[0]["model_id"] == "KV Cache Locality"
        assert data[0]["tldr_local_category"] == "model_analysis"

        def test_parse_tldr_newsletter_keeps_non_trending_items(self, monkeypatch):
                monkeypatch.setattr(
                        tldr_mod,
                        "_classify_item",
                        lambda *a, **k: (False, "model_analysis"),
                )

                html = """
                <section>
                    <header><h3>Deep Dives & Analysis</h3></header>
                    <article>
                        <a class="font-bold" href="https://example.com/deep-dive">
                            <h3>A Deep Model Analysis (4 minute read)</h3>
                        </a>
                        <div class="newsletter-html">Non-trending but relevant analysis.</div>
                    </article>
                </section>
                """
                data = tldr_mod._parse_tldr_newsletter(html)

                assert len(data) == 1
                assert data[0]["include_in_trending"] is False
                assert data[0]["tldr_local_category"] == "model_analysis"
    def test_merge_with_cached_tldr_items_deduplicates_by_url(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tldr_mod, "_TLDR_CACHE_PATH", str(tmp_path / "tldr_items.json"))
        monkeypatch.setattr(tldr_mod, "_TLDR_HISTORY_DAYS", 30)

        cached = [
            {
                "model_id": "Item A",
                "url": "https://example.com/a",
                "edition_date": "2026-05-01",
                "source": "tldr_ai",
            }
        ]
        tldr_mod._save_cached_tldr_items(cached)

        merged = tldr_mod._merge_with_cached_tldr_items(
            [
                {
                    "model_id": "Item A updated",
                    "url": "https://example.com/a",
                    "edition_date": "2026-05-02",
                    "source": "tldr_ai",
                }
            ]
        )

        assert len(merged) == 1
        assert merged[0]["model_id"] == "Item A updated"
        assert merged[0]["edition_date"] == "2026-05-02"

    def test_merge_with_cached_tldr_items_prunes_old_entries(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tldr_mod, "_TLDR_CACHE_PATH", str(tmp_path / "tldr_items.json"))
        monkeypatch.setattr(tldr_mod, "_TLDR_HISTORY_DAYS", 1)

        merged = tldr_mod._merge_with_cached_tldr_items(
            [
                {
                    "model_id": "Very old",
                    "url": "https://example.com/old",
                    "edition_date": "2000-01-01",
                    "source": "tldr_ai",
                },
                {
                    "model_id": "Recent",
                    "url": "https://example.com/recent",
                    "edition_date": "2099-01-01",
                    "source": "tldr_ai",
                },
            ]
        )

        assert any(item["model_id"] == "Recent" for item in merged)
        assert all(item["model_id"] != "Very old" for item in merged)


class TestTldrDateRangeFetching:
    def test_fetch_single_edition_success(self, monkeypatch):
        from datetime import date

        monkeypatch.setattr(tldr_mod, "_classify_item", lambda *a, **k: (True, "trending_new_models"))

        tldr_html = """
        <section>
          <header><h3>Headlines & Launches</h3></header>
          <article>
            <a class="font-bold" href="https://example.com/grok">
              <h3>xAI Launched Grok 4.3</h3>
            </a>
            <div class="newsletter-html">A powerful new model.</div>
          </article>
        </section>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = tldr_html
        mock_resp.raise_for_status.return_value = None

        with patch("llmwatch.agents.watchers.tldr_ai.requests.get", return_value=mock_resp):
            agent = tldr_mod.TLDRAIWatcher()
            result = agent._fetch_single_edition(date(2026, 5, 1))

        assert result is not None
        assert len(result) == 1
        assert result[0]["model_id"] == "xAI Launched Grok 4.3"
        assert result[0]["edition_date"] == "2026-05-01"

    def test_fetch_single_edition_not_published(self):
        from datetime import date

        mock_resp = MagicMock()
        mock_resp.status_code = 302

        with patch("llmwatch.agents.watchers.tldr_ai.requests.get", return_value=mock_resp):
            agent = tldr_mod.TLDRAIWatcher()
            result = agent._fetch_single_edition(date(2026, 5, 1))

        assert result is None

    def test_fetch_single_edition_network_error(self, monkeypatch):
        from datetime import date

        monkeypatch.setattr(
            tldr_mod.requests,
            "get",
            lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("Network error")),
        )

        agent = tldr_mod.TLDRAIWatcher()
        result = agent._fetch_single_edition(date(2026, 5, 1))

        assert result is None

    def test_run_with_date_range(self, monkeypatch, tmp_path):
        from datetime import date

        monkeypatch.setattr(tldr_mod, "_TLDR_CACHE_PATH", str(tmp_path / "tldr_items.json"))
        monkeypatch.setattr(tldr_mod, "_classify_item", lambda *a, **k: (True, "trending_new_models"))

        def _make_html(title, url):
            return f"""
            <section>
              <header><h3>Headlines & Launches</h3></header>
              <article>
                <a class="font-bold" href="{url}">
                  <h3>{title}</h3>
                </a>
                <div class="newsletter-html">A powerful new model.</div>
              </article>
            </section>
            """

        responses_by_url = {
            "https://tldr.tech/ai/2026-05-01": (200, _make_html("xAI Launched Grok 4.3", "https://example.com/grok-1")),
            "https://tldr.tech/ai/2026-05-02": (200, _make_html("OpenAI Launched GPT-5", "https://example.com/grok-2")),
        }

        def _fake_get(url, **kwargs):
            status, text = responses_by_url.get(url, (404, ""))
            mock_resp = MagicMock()
            mock_resp.status_code = status
            mock_resp.text = text
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        with patch("llmwatch.agents.watchers.tldr_ai.requests.get", side_effect=_fake_get):
            agent = tldr_mod.TLDRAIWatcher()
            result = agent.run(context={"date_range": (date(2026, 5, 1), date(2026, 5, 2))})

        assert not result.errors
        assert len(result.data) == 2
        assert any(item["edition_date"] == "2026-05-01" for item in result.data)
        assert any(item["edition_date"] == "2026-05-02" for item in result.data)

    def test_run_with_date_range_partial_data(self, monkeypatch, tmp_path):
        from datetime import date

        monkeypatch.setattr(tldr_mod, "_TLDR_CACHE_PATH", str(tmp_path / "tldr_items.json"))
        monkeypatch.setattr(tldr_mod, "_classify_item", lambda *a, **k: (True, "trending_new_models"))

        tldr_html = """
        <section>
          <header><h3>Headlines & Launches</h3></header>
          <article>
            <a class="font-bold" href="https://example.com/item">
              <h3>Test Item</h3>
            </a>
            <div class="newsletter-html">Test data.</div>
          </article>
        </section>
        """

        # Only May 1 is published; May 2 and 3 redirect (not published)
        def _fake_get(url, **kwargs):
            mock_resp = MagicMock()
            if url == "https://tldr.tech/ai/2026-05-01":
                mock_resp.status_code = 200
                mock_resp.text = tldr_html
            else:
                mock_resp.status_code = 302
                mock_resp.text = ""
            mock_resp.raise_for_status.return_value = None
            return mock_resp

        with patch("llmwatch.agents.watchers.tldr_ai.requests.get", side_effect=_fake_get):
            agent = tldr_mod.TLDRAIWatcher()
            result = agent.run(context={"date_range": (date(2026, 5, 1), date(2026, 5, 3))})

        assert not result.errors
        assert len(result.data) >= 1
        assert any(item["edition_date"] == "2026-05-01" for item in result.data)