"""
Unit tests for the watcher agents using mocked HTTP responses.
"""

import json

import pytest
import requests
import responses as resp_lib

from llmwatch.agents.watchers.huggingface import HuggingFaceTrendingWatcher, _extract_urls_from_card
from llmwatch.agents.watchers.ollama import OllamaModelWatcher, _parse_ollama_library


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
      <a href="/llama3.2">
        <h2>llama3.2</h2>
        <p class="description">Meta's Llama 3.2 model collection</p>
        <span>3B</span>
        <span>11B</span>
      </a>
    </li>
    <li>
      <a href="/mistral">
        <h2>mistral</h2>
        <p class="description">The Mistral 7B model</p>
        <span>7B</span>
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
            "https://ollama.com/library",
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
            "https://ollama.com/library",
            body=requests.ConnectionError("timeout"),
        )
        agent = OllamaModelWatcher()
        result = agent.run()

        assert not result.ok()
        assert result.data == []

    def test_parse_ollama_library_skips_nav_paths(self):
        html = '<a href="/library">Library</a><a href="/llama3.2">Llama 3.2</a>'
        data = _parse_ollama_library(html)
        model_ids = [d["model_id"] for d in data]
        assert "library" not in model_ids

    def test_parse_ollama_library_deduplicates(self):
        html = (
            '<a href="/llama3.2">Llama</a>'
            '<a href="/llama3.2">Llama duplicate</a>'
        )
        data = _parse_ollama_library(html)
        model_ids = [d["model_id"] for d in data]
        assert model_ids.count("llama3.2") <= 1

    def test_parse_ollama_library_sets_source(self):
        html = '<a href="/gemma3">Gemma 3</a>'
        data = _parse_ollama_library(html)
        for item in data:
            assert item["source"] == "ollama"
            assert item["url"].startswith("https://ollama.com/")
