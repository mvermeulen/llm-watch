"""
Unit tests for the arXiv lookup agent.
"""

import requests
import responses as resp_lib

from llmwatch.agents.base import AgentResult
from llmwatch.agents.lookup.arxiv import ArxivLookupAgent, _extract_search_terms, _parse_atom_feed

# Minimal Atom XML that mimics an arXiv API response
ARXIV_ATOM = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Efficient Llama Fine-Tuning at Scale</title>
    <summary>We present a method for efficient fine-tuning of large language models
using parameter-efficient techniques.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <link rel="alternate" href="https://arxiv.org/abs/2401.00001"/>
    <author><name>Jane Smith</name></author>
    <author><name>John Doe</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Scaling Laws for Neural Language Models</title>
    <summary>An empirical study of scaling laws for language models.</summary>
    <published>2024-01-16T00:00:00Z</published>
    <link rel="alternate" href="https://arxiv.org/abs/2401.00002"/>
    <author><name>Alice Brown</name></author>
  </entry>
</feed>
"""


class TestArxivLookupAgent:
    @resp_lib.activate
    def test_successful_lookup_with_context(self):
        resp_lib.add(
            resp_lib.GET,
            "https://export.arxiv.org/api/query",
            body=ARXIV_ATOM,
            status=200,
        )
        context = {
            "watcher_results": [
                AgentResult(
                    agent_name="huggingface_trending",
                    category="watcher",
                    data=[{"model_id": "meta-llama/Llama-3-8B", "tags": []}],
                )
            ]
        }
        agent = ArxivLookupAgent()
        agent.max_terms = 1
        result = agent.run(context=context)

        assert result.agent_name == "arxiv_lookup"
        assert result.category == "lookup"
        assert len(result.data) >= 1
        paper = result.data[0]
        assert "title" in paper
        assert "url" in paper
        assert "authors" in paper
        assert "published" in paper

    @resp_lib.activate
    def test_fallback_default_query_when_no_context(self):
        resp_lib.add(
            resp_lib.GET,
            "https://export.arxiv.org/api/query",
            body=ARXIV_ATOM,
            status=200,
        )
        agent = ArxivLookupAgent()
        agent.max_terms = 1
        result = agent.run(context=None)

        assert result.agent_name == "arxiv_lookup"
        # Even without context we still get results
        assert isinstance(result.data, list)

    @resp_lib.activate
    def test_network_error_recorded_in_errors(self):
        resp_lib.add(
            resp_lib.GET,
            "https://export.arxiv.org/api/query",
            body=requests.ConnectionError("connection error"),
        )
        context = {
            "watcher_results": [
                AgentResult(
                    agent_name="huggingface_trending",
                    category="watcher",
                    data=[{"model_id": "meta-llama/Llama-3-8B", "tags": []}],
                )
            ]
        }
        agent = ArxivLookupAgent()
        agent.max_terms = 1
        result = agent.run(context=context)

        assert len(result.errors) >= 1

    def test_deduplicates_papers(self):
        resp_lib.start()
        try:
            resp_lib.add(
                resp_lib.GET,
                "https://export.arxiv.org/api/query",
                body=ARXIV_ATOM,
                status=200,
            )
            resp_lib.add(
                resp_lib.GET,
                "https://export.arxiv.org/api/query",
                body=ARXIV_ATOM,
                status=200,
            )
            context = {
                "watcher_results": [
                    AgentResult(
                        agent_name="huggingface_trending",
                        category="watcher",
                        data=[
                            {"model_id": "meta-llama/Llama-3-8B", "tags": []},
                            {"model_id": "google/Gemma-2-9B", "tags": []},
                        ],
                    )
                ]
            }
            agent = ArxivLookupAgent()
            agent.max_terms = 2
            result = agent.run(context=context)
            urls = [p["url"] for p in result.data]
            assert len(urls) == len(set(urls)), "Duplicate URLs should be removed"
        finally:
            resp_lib.stop()
            resp_lib.reset()


class TestExtractSearchTerms:
    def test_strips_org_prefix(self):
        ctx = {
            "watcher_results": [
                AgentResult(
                    agent_name="hf",
                    category="watcher",
                    data=[{"model_id": "mistralai/Mistral-7B-Instruct-v0.2", "tags": []}],
                )
            ]
        }
        terms = _extract_search_terms(ctx)
        # Should NOT contain "mistralai/" prefix
        assert all("/" not in t for t in terms)
        assert any("Mistral" in t for t in terms)

    def test_no_context_returns_empty(self):
        assert _extract_search_terms(None) == []
        assert _extract_search_terms({}) == []

    def test_deduplicates_terms(self):
        ctx = {
            "watcher_results": [
                AgentResult(
                    agent_name="hf",
                    category="watcher",
                    data=[
                        {"model_id": "org/Llama3", "tags": []},
                        {"model_id": "other/Llama3", "tags": []},
                    ],
                )
            ]
        }
        terms = _extract_search_terms(ctx)
        assert terms.count("Llama3") <= 1


class TestParseAtomFeed:
    def test_parses_titles_and_authors(self):
        papers = _parse_atom_feed(ARXIV_ATOM, "llama")
        assert len(papers) == 2
        assert papers[0]["title"] == "Efficient Llama Fine-Tuning at Scale"
        assert "Jane Smith" in papers[0]["authors"]
        assert "John Doe" in papers[0]["authors"]

    def test_parses_published_date(self):
        papers = _parse_atom_feed(ARXIV_ATOM, "llama")
        assert papers[0]["published"] == "2024-01-15"

    def test_parses_url(self):
        papers = _parse_atom_feed(ARXIV_ATOM, "llama")
        assert papers[0]["url"] == "https://arxiv.org/abs/2401.00001"

    def test_summary_truncated(self):
        papers = _parse_atom_feed(ARXIV_ATOM, "llama")
        assert len(papers[0]["summary"]) <= 301  # 300 chars + possible ellipsis

    def test_stores_query_term(self):
        papers = _parse_atom_feed(ARXIV_ATOM, "my_query")
        for p in papers:
            assert p["query"] == "my_query"

    def test_empty_feed_returns_empty_list(self):
        empty_atom = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        )
        papers = _parse_atom_feed(empty_atom, "test")
        assert papers == []
