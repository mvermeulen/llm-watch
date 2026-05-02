"""
Unit tests for the watcher agents using mocked HTTP responses.
"""

import json

import pytest
import requests
import responses as resp_lib

from llmwatch.agents.watchers.huggingface import HuggingFaceTrendingWatcher, _extract_urls_from_card
from llmwatch.agents.watchers.ollama import OllamaModelWatcher, _parse_ollama_library
from llmwatch.agents.watchers import tldr_ai as tldr_mod


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
    @resp_lib.activate
    def test_successful_fetch(self):
        resp_lib.add(
            resp_lib.GET,
            "https://huggingface.co/api/models",
            json=HF_SAMPLE,
            status=200,
        )
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

    @resp_lib.activate
    def test_network_error_returns_error_result(self):
        resp_lib.add(
            resp_lib.GET,
            "https://huggingface.co/api/models",
            body=requests.ConnectionError("connection refused"),
        )
        agent = HuggingFaceTrendingWatcher()
        result = agent.run()

        assert not result.ok()
        assert result.data == []
        assert any("connection refused" in e for e in result.errors)

    @resp_lib.activate
    def test_http_error_returns_error_result(self):
        resp_lib.add(
            resp_lib.GET,
            "https://huggingface.co/api/models",
            status=500,
            json={"error": "Internal Server Error"},
        )
        agent = HuggingFaceTrendingWatcher()
        result = agent.run()

        assert not result.ok()

    @resp_lib.activate
    def test_new_sources_extracted_from_card_data(self):
        resp_lib.add(
            resp_lib.GET,
            "https://huggingface.co/api/models",
            json=HF_SAMPLE,
            status=200,
        )
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
    @resp_lib.activate
    def test_successful_scrape(self):
        resp_lib.add(
            resp_lib.GET,
            "https://ollama.com/search",
            body=OLLAMA_HTML,
            status=200,
        )
        agent = OllamaModelWatcher()
        result = agent.run()

        assert result.ok()
        assert result.agent_name == "ollama_models"
        assert result.category == "watcher"
        # At least some models should be found
        model_ids = [d["model_id"] for d in result.data]
        assert "llama3.2" in model_ids or "mistral" in model_ids

    @resp_lib.activate
    def test_network_error_returns_error_result(self):
        resp_lib.add(
            resp_lib.GET,
            "https://ollama.com/search",
            body=requests.ConnectionError("timeout"),
        )
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
        monkeypatch.setattr(tldr_mod, "_TLDR_CACHE_DIR", str(tmp_path))
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
        monkeypatch.setattr(tldr_mod, "_TLDR_CACHE_DIR", str(tmp_path))
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
    @resp_lib.activate
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
        resp_lib.add(
            resp_lib.GET,
            "https://tldr.tech/ai/2026-05-01",
            body=tldr_html,
            status=200,
        )

        agent = tldr_mod.TLDRAIWatcher()
        result = agent._fetch_single_edition(date(2026, 5, 1))

        assert result is not None
        assert len(result) == 1
        assert result[0]["model_id"] == "xAI Launched Grok 4.3"
        assert result[0]["edition_date"] == "2026-05-01"

    @resp_lib.activate
    def test_fetch_single_edition_not_published(self):
        from datetime import date

        resp_lib.add(
            resp_lib.GET,
            "https://tldr.tech/ai/2026-05-01",
            status=302,
        )

        agent = tldr_mod.TLDRAIWatcher()
        result = agent._fetch_single_edition(date(2026, 5, 1))

        assert result is None

    @resp_lib.activate
    def test_fetch_single_edition_network_error(self, monkeypatch):
        from datetime import date

        def raise_error(*args, **kwargs):
            raise requests.ConnectionError("Network error")

        monkeypatch.setattr(tldr_mod.requests, "get", raise_error)

        agent = tldr_mod.TLDRAIWatcher()
        result = agent._fetch_single_edition(date(2026, 5, 1))

        assert result is None

    @resp_lib.activate
    def test_run_with_date_range(self, monkeypatch, tmp_path):
        from datetime import date

        monkeypatch.setattr(tldr_mod, "_TLDR_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(tldr_mod, "_TLDR_CACHE_PATH", str(tmp_path / "tldr_items.json"))
        monkeypatch.setattr(tldr_mod, "_classify_item", lambda *a, **k: (True, "trending_new_models"))

        tldr_html_may_01 = """
        <section>
          <header><h3>Headlines & Launches</h3></header>
          <article>
            <a class="font-bold" href="https://example.com/grok-1">
              <h3>xAI Launched Grok 4.3</h3>
            </a>
            <div class="newsletter-html">A powerful new model.</div>
          </article>
        </section>
        """

        tldr_html_may_02 = """
        <section>
          <header><h3>Headlines & Launches</h3></header>
          <article>
            <a class="font-bold" href="https://example.com/grok-2">
              <h3>OpenAI Launched GPT-5</h3>
            </a>
            <div class="newsletter-html">Another new model.</div>
          </article>
        </section>
        """

        resp_lib.add(
            resp_lib.GET,
            "https://tldr.tech/ai/2026-05-01",
            body=tldr_html_may_01,
            status=200,
        )
        resp_lib.add(
            resp_lib.GET,
            "https://tldr.tech/ai/2026-05-02",
            body=tldr_html_may_02,
            status=200,
        )

        agent = tldr_mod.TLDRAIWatcher()
        result = agent.run(context={"date_range": (date(2026, 5, 1), date(2026, 5, 2))})

        assert not result.errors
        assert len(result.data) == 2
        assert any(item["edition_date"] == "2026-05-01" for item in result.data)
        assert any(item["edition_date"] == "2026-05-02" for item in result.data)

    @resp_lib.activate
    def test_run_with_date_range_partial_data(self, monkeypatch, tmp_path):
        from datetime import date

        monkeypatch.setattr(tldr_mod, "_TLDR_CACHE_DIR", str(tmp_path))
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
        resp_lib.add(
            resp_lib.GET,
            "https://tldr.tech/ai/2026-05-01",
            body=tldr_html,
            status=200,
        )
        resp_lib.add(
            resp_lib.GET,
            "https://tldr.tech/ai/2026-05-02",
            status=302,
        )
        resp_lib.add(
            resp_lib.GET,
            "https://tldr.tech/ai/2026-05-03",
            status=302,
        )

        agent = tldr_mod.TLDRAIWatcher()
        result = agent.run(context={"date_range": (date(2026, 5, 1), date(2026, 5, 3))})

        assert not result.errors
        assert len(result.data) >= 1
        assert any(item["edition_date"] == "2026-05-01" for item in result.data)