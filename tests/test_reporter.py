"""
Unit tests for the weekly reporter agent.
"""

from llmwatch.agents.base import AgentResult
from llmwatch.agents.reporter import WeeklyReporterAgent, _collect_new_sources, _source_label


def _make_watcher_result(agent_name: str, models: list[dict]) -> AgentResult:
    return AgentResult(
        agent_name=agent_name,
        category="watcher",
        data=models,
    )


def _make_lookup_result(papers: list[dict]) -> AgentResult:
    return AgentResult(
        agent_name="arxiv_lookup",
        category="lookup",
        data=papers,
    )


class TestWeeklyReporterAgent:
    def test_generates_markdown_report(self):
        agent = WeeklyReporterAgent()
        ctx = {
            "watcher_results": [
                _make_watcher_result(
                    "huggingface_trending",
                    [
                        {
                            "model_id": "mistralai/Mistral-7B",
                            "url": "https://huggingface.co/mistralai/Mistral-7B",
                            "description": "A fast model.",
                            "tags": ["text-generation"],
                            "source": "huggingface",
                        }
                    ],
                )
            ],
            "lookup_results": [
                _make_lookup_result(
                    [
                        {
                            "title": "Mistral: A Powerful LLM",
                            "url": "https://arxiv.org/abs/1234.56789",
                            "authors": "Doe, J.",
                            "published": "2024-01-01",
                            "summary": "Abstract text.",
                            "query": "Mistral",
                        }
                    ]
                )
            ],
        }
        result = agent.run(context=ctx)

        assert result.ok()
        assert result.agent_name == "weekly_reporter"
        assert len(result.data) == 1
        report = result.data[0]["report"]
        assert "# LLM Watch" in report
        assert "Mistral-7B" in report
        assert "Mistral: A Powerful LLM" in report
        assert "date" in result.data[0]

    def test_report_with_no_context(self):
        agent = WeeklyReporterAgent()
        result = agent.run(context={})
        assert result.ok()
        report = result.data[0]["report"]
        assert "No watcher data available" in report
        assert "No papers retrieved" in report

    def test_report_includes_errors_section(self):
        agent = WeeklyReporterAgent()
        ctx = {
            "watcher_results": [
                AgentResult(
                    agent_name="huggingface_trending",
                    category="watcher",
                    data=[],
                    errors=["API timed out"],
                )
            ],
            "lookup_results": [],
        }
        result = agent.run(context=ctx)
        report = result.data[0]["report"]
        assert "Warnings" in report
        assert "API timed out" in report

    def test_report_includes_new_sources_section(self):
        agent = WeeklyReporterAgent()
        ctx = {
            "watcher_results": [
                AgentResult(
                    agent_name="huggingface_trending",
                    category="watcher",
                    data=[
                        {
                            "model_id": "test/model",
                            "url": "https://huggingface.co/test/model",
                            "description": "See https://newblog.example.com/post",
                            "tags": [],
                            "source": "huggingface",
                        }
                    ],
                    new_sources=["https://newblog.example.com/post"],
                )
            ],
            "lookup_results": [],
        }
        result = agent.run(context=ctx)
        report = result.data[0]["report"]
        assert "New Sources Discovered" in report
        assert "https://newblog.example.com/post" in report

    def test_report_caps_models_per_source(self):
        """Reporter should cap at 15 models per source."""
        agent = WeeklyReporterAgent()
        many_models = [
            {
                "model_id": f"org/model-{i}",
                "url": f"https://huggingface.co/org/model-{i}",
                "description": "",
                "tags": [],
                "source": "huggingface",
            }
            for i in range(30)
        ]
        ctx = {
            "watcher_results": [_make_watcher_result("huggingface_trending", many_models)],
            "lookup_results": [],
        }
        result = agent.run(context=ctx)
        report = result.data[0]["report"]
        # Count model lines
        model_lines = [l for l in report.splitlines() if "org/model-" in l]
        assert len(model_lines) <= 15

    def test_tldr_non_trending_routed_to_analysis_section(self):
        agent = WeeklyReporterAgent()
        ctx = {
            "watcher_results": [
                _make_watcher_result(
                    "tldr_ai",
                    [
                        {
                            "model_id": "KV Cache Locality",
                            "url": "https://example.com/kv-cache",
                            "description": "Latency and throughput deep dive.",
                            "tags": ["Deep Dives & Analysis"],
                            "source": "tldr_ai",
                            "include_in_trending": False,
                            "tldr_local_category": "model_analysis",
                        }
                    ],
                )
            ],
            "lookup_results": [],
        }

        result = agent.run(context=ctx)
        report = result.data[0]["report"]

        assert "TLDR Model Analysis" in report
        assert "KV Cache Locality" in report
        assert "Trending & New Models" in report
        trending_lines = [l for l in report.splitlines() if "KV Cache Locality" in l and l.startswith("- ")]
        assert len(trending_lines) == 1

    def test_tldr_non_trending_routed_to_other_news_section(self):
        agent = WeeklyReporterAgent()
        ctx = {
            "watcher_results": [
                _make_watcher_result(
                    "tldr_ai",
                    [
                        {
                            "model_id": "Anthropic Valuation Update",
                            "url": "https://example.com/valuation",
                            "description": "Funding and valuation round details.",
                            "tags": ["Headlines & Launches"],
                            "source": "tldr_ai",
                            "include_in_trending": False,
                            "tldr_local_category": "other",
                        }
                    ],
                )
            ],
            "lookup_results": [],
        }

        result = agent.run(context=ctx)
        report = result.data[0]["report"]

        assert "TLDR Other AI News" in report
        assert "Anthropic Valuation Update" in report

    def test_report_includes_lastweekinai_sections(self):
        agent = WeeklyReporterAgent()
        ctx = {
            "watcher_results": [
                _make_watcher_result(
                    "lastweekinai_podcast",
                    [
                        {
                            "model_id": "LWiAI Podcast #242 - Test",
                            "url": "https://lastweekin.ai/p/lwiai-podcast-242-test",
                            "description": "Weekly summary",
                            "tags": ["podcast_summary"],
                            "source": "lastweekinai_podcast",
                            "published": "2026-04-30",
                        },
                        {
                            "model_id": "Example Source",
                            "url": "https://example.com/story",
                            "description": "Referenced in podcast",
                            "tags": ["podcast_link", "example.com"],
                            "source": "lastweekinai_podcast",
                            "episode_title": "LWiAI Podcast #242 - Test",
                            "published": "2026-04-30",
                        },
                    ],
                )
            ],
            "lookup_results": [],
        }

        result = agent.run(context=ctx)
        report = result.data[0]["report"]

        assert "Last Week in AI Podcast Summaries" in report
        assert "Last Week in AI Referenced Links" in report
        assert "Example Source" in report

    def test_report_includes_dedicated_neuron_summaries_section(self):
        agent = WeeklyReporterAgent()
        ctx = {
            "watcher_results": [
                _make_watcher_result(
                    "neuron_feed",
                    [
                        {
                            "model_id": "Around the Horn Digest",
                            "url": "https://www.theneuron.ai/explainer-articles/test/",
                            "description": "A roundup of the day in AI.",
                            "tags": ["neuron", "explainer-articles"],
                            "source": "neuron",
                            "published": "2026-05-01",
                            "neuron_category": "explainer-articles",
                        }
                    ],
                )
            ],
            "lookup_results": [],
        }

        result = agent.run(context=ctx)
        report = result.data[0]["report"]

        assert "The Neuron Summaries" in report
        assert "Around the Horn Digest" in report
        assert "2026-05-01" in report
        assert "explainer-articles" in report

    def test_report_includes_common_links_section_and_ranks_by_signal(self):
        agent = WeeklyReporterAgent()
        ctx = {
            "watcher_results": [],
            "lookup_results": [],
            "consolidated_stories": [
                {
                    "primary_item": {
                        "model_id": "Single Source Repeated Link",
                        "url": "https://example.com/repeated",
                        "description": "Repeated mentions from one source.",
                    },
                    "appearances": [
                        {"source": "neuron_feed", "date": "2026-05-01", "url": "https://example.com/repeated"},
                        {"source": "neuron_feed", "date": "2026-05-01", "url": "https://example.com/repeated"},
                        {"source": "neuron_feed", "date": "2026-05-01", "url": "https://example.com/repeated"},
                    ],
                    "impact_score": 3,
                    "source_count": 1,
                    "common_link_type": "social_post",
                    "common_link_signal": 13,
                },
                {
                    "primary_item": {
                        "model_id": "Cross Source Common Link",
                        "url": "https://example.com/cross-source",
                        "description": "Referenced across different feeds.",
                    },
                    "appearances": [
                        {"source": "tldr_ai", "date": "2026-05-01", "url": "https://example.com/cross-source"},
                        {"source": "lastweekinai_podcast", "date": "2026-05-01", "url": "https://example.com/cross-source"},
                    ],
                    "impact_score": 2,
                    "source_count": 2,
                    "common_link_type": "news_story",
                    "common_link_signal": 22,
                },
            ],
        }

        result = agent.run(context=ctx)
        report = result.data[0]["report"]

        assert "Common Links This Week" in report
        assert "cross-source signal" in report
        assert "High Signal Common Links" in report
        assert "Repeated References" in report
        assert "Type" in report

        assert report.index("Cross Source Common Link") < report.index("Single Source Repeated Link")

    def test_report_shows_suppressed_links_in_summary_only(self):
        agent = WeeklyReporterAgent()
        ctx = {
            "watcher_results": [],
            "lookup_results": [],
            "consolidated_stories": [
                {
                    "primary_item": {
                        "model_id": "Visible Link",
                        "url": "https://example.com/visible",
                        "description": "Visible",
                    },
                    "appearances": [
                        {"source": "tldr_ai", "date": "2026-05-01", "url": "https://example.com/visible"},
                        {"source": "lastweekinai_podcast", "date": "2026-05-01", "url": "https://example.com/visible"},
                    ],
                    "impact_score": 2,
                    "source_count": 2,
                    "common_link_type": "news_story",
                    "common_link_signal": 20,
                    "suppressed": False,
                    "suppression_reason": "",
                },
                {
                    "primary_item": {
                        "model_id": "Suppressed Sponsor",
                        "url": "https://example.com/sponsor",
                        "description": "Sponsored",
                    },
                    "appearances": [
                        {"source": "tldr_ai", "date": "2026-05-01", "url": "https://example.com/sponsor"},
                    ],
                    "impact_score": 1,
                    "source_count": 1,
                    "common_link_type": "sponsor",
                    "common_link_signal": 1,
                    "suppressed": True,
                    "suppression_reason": "sponsor_link",
                },
            ],
        }

        result = agent.run(context=ctx)
        report = result.data[0]["report"]

        assert "Suppressed Repeated Links" in report
        assert "Suppressed Sponsor" in report
        assert "(sponsor)" in report
        assert report.index("Visible Link") < report.index("Suppressed Sponsor")


class TestSourceLabel:
    def test_known_agents(self):
        assert "HuggingFace" in _source_label("huggingface_trending")
        assert "Ollama" in _source_label("ollama_models")

    def test_unknown_agent_falls_back_to_title_case(self):
        label = _source_label("my_custom_agent")
        assert label == "My Custom Agent"


class TestCollectNewSources:
    def test_extracts_external_urls(self):
        results = [
            AgentResult(
                agent_name="hf",
                category="watcher",
                data=[{"description": "Check https://coolsite.example.com/info"}],
                new_sources=["https://another.example.net"],
            )
        ]
        sources = _collect_new_sources(results)
        assert "https://coolsite.example.com/info" in sources
        assert set(sources) >= {"https://another.example.net"}

    def test_filters_known_domains(self):
        results = [
            AgentResult(
                agent_name="hf",
                category="watcher",
                data=[{"url": "https://huggingface.co/test"}],
                new_sources=["https://github.com/repo"],
            )
        ]
        sources = _collect_new_sources(results)
        # huggingface.co and github.com should be filtered out
        assert "https://huggingface.co/test" not in sources
        assert "https://github.com/repo" not in sources

    def test_deduplicates(self):
        url = "https://unique.example.com/page"
        results = [
            AgentResult(
                agent_name="hf",
                category="watcher",
                data=[{"description": url}, {"description": url}],
                new_sources=[url],
            )
        ]
        sources = _collect_new_sources(results)
        assert sources.count(url) == 1
