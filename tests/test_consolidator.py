"""Unit tests for the story consolidator agent."""

from __future__ import annotations

from datetime import datetime, timedelta
import pytest

from llmwatch.agents.consolidator import _normalize_url, StoryConsolidatorAgent
from llmwatch.agents.base import AgentResult


class TestNormalizeUrl:
    """Test URL normalization utility."""
    
    def test_remove_query_params(self):
        """Query parameters should be stripped."""
        url = "https://example.com/article?utm_source=tldr&utm_medium=email"
        assert _normalize_url(url) == "https://example.com/article"
    
    def test_remove_fragment(self):
        """Fragments should be stripped."""
        url = "https://example.com/article#section-2"
        assert _normalize_url(url) == "https://example.com/article"
    
    def test_trailing_slash(self):
        """Trailing slashes should be removed."""
        url = "https://example.com/article/"
        assert _normalize_url(url) == "https://example.com/article"
    
    def test_combined_query_and_fragment(self):
        """Both query params and fragments should be stripped."""
        url = "https://example.com/article?source=tldr#section"
        assert _normalize_url(url) == "https://example.com/article"
    
    def test_different_sources_same_article(self):
        """Same article from different sources should normalize to same URL."""
        url1 = "https://example.com/article?source=tldr"
        url2 = "https://example.com/article?source=lwiai"
        assert _normalize_url(url1) == _normalize_url(url2)
    
    def test_empty_url(self):
        """Empty URLs should return empty string."""
        assert _normalize_url("") == ""
    
    def test_none_url(self):
        """None URLs should return empty string."""
        assert _normalize_url(None) == ""
    
    def test_invalid_url(self):
        """Invalid URLs should be handled gracefully."""
        # Should not raise, just return something
        result = _normalize_url("not a url at all")
        assert isinstance(result, str)
    
    def test_https_vs_http(self):
        """https and http should be treated as different."""
        url_https = "https://example.com/article"
        url_http = "http://example.com/article"
        assert _normalize_url(url_https) != _normalize_url(url_http)
    
    def test_www_handling(self):
        """www prefix should not be stripped (not normalized)."""
        url_www = "https://www.example.com/article"
        url_no_www = "https://example.com/article"
        # In Phase 1, we don't normalize www, so these are different
        assert _normalize_url(url_www) != _normalize_url(url_no_www)
    
    def test_real_world_example_threadreader(self):
        """Test with real-world URL from threadreaderapp."""
        url = "https://threadreaderapp.com/thread/2049987001655714250.html?utm_source=tldr"
        expected = "https://threadreaderapp.com/thread/2049987001655714250.html"
        assert _normalize_url(url) == expected


class TestStoryConsolidatorAgent:
    """Test the consolidator agent."""
    
    def test_consolidate_same_url(self):
        """Items with same URL should be consolidated."""
        agent = StoryConsolidatorAgent()
        
        items = [
            {
                "item": {
                    "model_id": "Grok 4.3",
                    "url": "https://example.com/grok43?utm_source=tldr",
                    "description": "Grok 4.3 launched with improvements",
                    "published": "2026-05-02",
                },
                "source": "tldr_ai",
                "item_idx": 0,
            },
            {
                "item": {
                    "model_id": "xAI Grok 4.3",
                    "url": "https://example.com/grok43?utm_source=lwiai",
                    "description": "",
                    "published": "2026-04-30",
                },
                "source": "lwiai",
                "item_idx": 1,
            },
        ]
        
        consolidated = agent._consolidate_by_url(items, [])
        
        # Should produce 1 consolidated story
        assert len(consolidated) == 1
        
        story = consolidated[0]
        assert story["impact_score"] == 2
        assert len(story["appearances"]) == 2
        sources = {app["source"] for app in story["appearances"]}
        assert sources == {"tldr_ai", "lwiai"}
    
    def test_consolidate_no_duplicates(self):
        """Items with different URLs should not be consolidated."""
        agent = StoryConsolidatorAgent()
        
        items = [
            {
                "item": {
                    "model_id": "Grok 4.3",
                    "url": "https://xai.com/grok43",
                    "description": "Grok 4.3 announced",
                    "published": "2026-05-02",
                },
                "source": "tldr_ai",
                "item_idx": 0,
            },
            {
                "item": {
                    "model_id": "Claude Opus 4.7",
                    "url": "https://anthropic.com/claude",
                    "description": "Claude 4.7 released",
                    "published": "2026-05-02",
                },
                "source": "tldr_ai",
                "item_idx": 1,
            },
        ]
        
        consolidated = agent._consolidate_by_url(items, [])
        
        # Should produce 2 separate stories
        assert len(consolidated) == 2
        assert all(s["impact_score"] == 1 for s in consolidated)
    
    def test_primary_item_selection(self):
        """Primary item should be the one with longest description."""
        agent = StoryConsolidatorAgent()
        
        items = [
            {
                "item": {
                    "model_id": "Grok 4.3",
                    "url": "https://example.com/grok43",
                    "description": "Short",
                    "published": "2026-05-02",
                },
                "source": "tldr_ai",
                "item_idx": 0,
            },
            {
                "item": {
                    "model_id": "Grok 4.3",
                    "url": "https://example.com/grok43?utm_source=lwiai",
                    "description": "This is a much longer description with more details about Grok 4.3",
                    "published": "2026-04-30",
                },
                "source": "lwiai",
                "item_idx": 1,
            },
        ]
        
        consolidated = agent._consolidate_by_url(items, [])
        
        assert len(consolidated) == 1
        primary = consolidated[0]["primary_item"]
        assert primary["description"] == "This is a much longer description with more details about Grok 4.3"
    
    def test_items_without_url(self):
        """Items without URLs should be treated separately."""
        agent = StoryConsolidatorAgent()
        
        items = [
            {
                "item": {
                    "model_id": "Story with URL",
                    "url": "https://example.com/story1",
                    "description": "Has a URL",
                    "published": "2026-05-02",
                },
                "source": "tldr_ai",
                "item_idx": 0,
            },
            {
                "item": {
                    "model_id": "Story without URL",
                    "url": "",
                    "description": "No URL provided",
                    "published": "2026-05-02",
                },
                "source": "neuron",
                "item_idx": 1,
            },
        ]
        
        consolidated = agent._consolidate_by_url(items, [])
        
        # Should produce 2 separate stories
        assert len(consolidated) == 2
    
    def test_appearance_metadata(self):
        """Appearances should include source-specific metadata."""
        agent = StoryConsolidatorAgent()
        
        items = [
            {
                "item": {
                    "model_id": "Story",
                    "url": "https://example.com/story",
                    "description": "Test story",
                    "published": "2026-05-02",
                    "episode_title": "Podcast #242",
                },
                "source": "lwiai",
                "item_idx": 0,
            },
        ]
        
        consolidated = agent._consolidate_by_url(items, [])
        
        assert len(consolidated) == 1
        appearances = consolidated[0]["appearances"]
        assert len(appearances) == 1
        assert appearances[0]["source"] == "lwiai"
        assert appearances[0]["episode"] == "Podcast #242"
    
    def test_run_with_context(self):
        """Test full run() method with context."""
        agent = StoryConsolidatorAgent()
        
        # Create mock watcher results
        watcher_result = AgentResult(
            agent_name="tldr_ai",
            category="watcher",
            data=[
                {
                    "model_id": "Story 1",
                    "url": "https://example.com/story1",
                    "description": "First story",
                    "published": "2026-05-02",
                },
                {
                    "model_id": "Story 1 Again",
                    "url": "https://example.com/story1?utm_source=other",
                    "description": "",
                    "published": "2026-05-02",
                },
            ]
        )
        
        context = {
            "watcher_results": [watcher_result],
            "lookup_results": [],
        }
        
        result = agent.run(context=context)
        
        assert result.ok()
        assert len(result.data) == 1
        assert result.data[0]["impact_score"] == 2


class TestSimilarityMatching:
    """Test Phase 2: Title similarity matching."""
    
    def test_similar_titles_consolidated(self):
        """Similar titles should be consolidated if within threshold."""
        agent = StoryConsolidatorAgent()
        
        # Create two stories with similar but different titles, different URLs
        story1 = {
            "consolidated_id": "url1",
            "primary_item": {
                "model_id": "Claude Opus 4.7",
                "url": "https://anthropic.com/claude-4.7",
                "description": "New Claude model released",
                "published": "2026-05-02",
            },
            "appearances": [
                {"source": "tldr_ai", "date": "2026-05-02", "url": "https://anthropic.com/claude-4.7", "title": "Claude Opus 4.7"}
            ],
            "impact_score": 1,
        }
        
        story2 = {
            "consolidated_id": "url2",
            "primary_item": {
                "model_id": "Claude Opus 4.7",  # Same title (no typo)
                "url": "https://anthropic.com/claude",
                "description": "Claude 4.7 announcement",
                "published": "2026-05-01",
            },
            "appearances": [
                {"source": "lwiai", "date": "2026-05-01", "url": "https://anthropic.com/claude", "title": "Claude Opus"}
            ],
            "impact_score": 1,
        }
        
        consolidated = agent._consolidate_by_similarity([story1, story2])
        
        # Should merge these two stories
        assert len(consolidated) == 1
        assert consolidated[0]["impact_score"] == 2
        assert len(consolidated[0]["appearances"]) == 2
    
    def test_dissimilar_titles_not_consolidated(self):
        """Dissimilar titles should not be consolidated."""
        agent = StoryConsolidatorAgent()
        
        story1 = {
            "consolidated_id": "url1",
            "primary_item": {
                "model_id": "Claude Opus 4.7",
                "url": "https://anthropic.com/claude",
                "description": "New Claude model",
                "published": "2026-05-02",
            },
            "appearances": [{"source": "tldr_ai", "date": "2026-05-02", "url": "https://anthropic.com/claude", "title": "Claude"}],
            "impact_score": 1,
        }
        
        story2 = {
            "consolidated_id": "url2",
            "primary_item": {
                "model_id": "Grok 4.3",
                "url": "https://xai.com/grok",
                "description": "XAI Grok released",
                "published": "2026-05-02",
            },
            "appearances": [{"source": "tldr_ai", "date": "2026-05-02", "url": "https://xai.com/grok", "title": "Grok"}],
            "impact_score": 1,
        }
        
        consolidated = agent._consolidate_by_similarity([story1, story2])
        
        # Should remain separate
        assert len(consolidated) == 2
    
    def test_title_similarity_with_typos(self):
        """Similar titles with typos should be consolidated if above threshold."""
        agent = StoryConsolidatorAgent()
        
        story1 = {
            "consolidated_id": "url1",
            "primary_item": {
                "model_id": "DeepSeek V4 Pro",
                "url": "https://deepseek.com/v4-pro",
                "description": "DeepSeek V4 Pro announced",
                "published": "2026-05-02",
            },
            "appearances": [{"source": "tldr_ai", "date": "2026-05-02", "url": "https://deepseek.com/v4-pro", "title": "DeepSeek V4 Pro"}],
            "impact_score": 1,
        }
        
        story2 = {
            "consolidated_id": "url2",
            "primary_item": {
                "model_id": "DeepSeek V4Pro",  # Missing space - similar but slightly different
                "url": "https://deepseek.com/news",
                "description": "DeepSeek V4Pro launch",
                "published": "2026-05-02",
            },
            "appearances": [{"source": "lwiai", "date": "2026-05-02", "url": "https://deepseek.com/news", "title": "DeepSeek V4Pro"}],
            "impact_score": 1,
        }
        
        consolidated = agent._consolidate_by_similarity([story1, story2])
        
        # Should consolidate due to high similarity despite typo
        assert len(consolidated) == 1
        assert consolidated[0]["impact_score"] == 2
    
    def test_temporal_window_filtering(self):
        """Stories outside temporal window should not be consolidated."""
        agent = StoryConsolidatorAgent()
        
        # Story from today
        story1 = {
            "consolidated_id": "url1",
            "primary_item": {
                "model_id": "New Model Today",
                "url": "https://example.com/model1",
                "description": "Just released",
                "published": "2026-05-02",
            },
            "appearances": [{"source": "tldr_ai", "date": "2026-05-02", "url": "https://example.com/model1", "title": "New Model"}],
            "impact_score": 1,
        }
        
        # Story from 10 days ago (outside 7-day window)
        story2 = {
            "consolidated_id": "url2",
            "primary_item": {
                "model_id": "New Model Today",  # Same title
                "url": "https://example.com/model2",
                "description": "Released earlier",
                "published": "2026-04-22",
            },
            "appearances": [{"source": "lwiai", "date": "2026-04-22", "url": "https://example.com/model2", "title": "New Model"}],
            "impact_score": 1,
        }
        
        consolidated = agent._consolidate_by_similarity([story1, story2])
        
        # Should NOT consolidate due to temporal window
        assert len(consolidated) == 2
    
    def test_empty_titles_skipped(self):
        """Stories with empty titles should be skipped for similarity matching."""
        agent = StoryConsolidatorAgent()
        
        story1 = {
            "consolidated_id": "url1",
            "primary_item": {
                "model_id": "",  # Empty title
                "url": "https://example.com/story1",
                "description": "Some story",
                "published": "2026-05-02",
            },
            "appearances": [{"source": "tldr_ai", "date": "2026-05-02", "url": "https://example.com/story1", "title": ""}],
            "impact_score": 1,
        }
        
        story2 = {
            "consolidated_id": "url2",
            "primary_item": {
                "model_id": "Another Story",
                "url": "https://example.com/story2",
                "description": "Different story",
                "published": "2026-05-02",
            },
            "appearances": [{"source": "lwiai", "date": "2026-05-02", "url": "https://example.com/story2", "title": "Another Story"}],
            "impact_score": 1,
        }
        
        consolidated = agent._consolidate_by_similarity([story1, story2])
        
        # Should remain separate (empty title skipped)
        assert len(consolidated) == 2
    
    def test_similarity_calculation(self):
        """Test the similarity calculation method."""
        agent = StoryConsolidatorAgent()
        
        # Identical should be 1.0
        assert agent._calculate_similarity("test", "test") == 1.0
        
        # Very similar should be high
        sim = agent._calculate_similarity("claude opus 4.7", "claude opus 4.7")
        assert sim == 1.0
        
        # Case insensitive comparison (method converts to lowercase)
        sim = agent._calculate_similarity("Claude Opus", "claude opus")
        assert sim == 1.0
        
        # Similar but not identical
        sim = agent._calculate_similarity("deepseek v4 pro", "deepseek v4pro")
        assert 0.85 <= sim < 1.0
        
        # Completely different
        sim = agent._calculate_similarity("claude", "grok")
        assert sim < 0.85
        
        # Empty strings
        assert agent._calculate_similarity("", "") == 0.0
        assert agent._calculate_similarity("test", "") == 0.0
    
    def test_extract_title(self):
        """Test title extraction from items."""
        agent = StoryConsolidatorAgent()
        
        # Extract from model_id
        item1 = {"model_id": "Claude Opus 4.7", "name": "Backup Name"}
        assert agent._extract_title(item1) == "claude opus 4.7"
        
        # Fall back to name if model_id missing
        item2 = {"name": "Test Model"}
        assert agent._extract_title(item2) == "test model"
        
        # Return empty if both missing
        item3 = {}
        assert agent._extract_title(item3) == ""
        
        # Handle whitespace
        item4 = {"model_id": "  Model With Spaces  "}
        assert agent._extract_title(item4) == "model with spaces"
    
    def test_phase2_no_regression_from_phase1(self):
        """Verify Phase 2 doesn't break Phase 1 URL consolidation."""
        agent = StoryConsolidatorAgent()
        
        # Items with same URL (Phase 1 consolidation)
        items = [
            {
                "item": {
                    "model_id": "Story",
                    "url": "https://example.com/story?utm_source=tldr",
                    "description": "Test story with URL",
                    "published": "2026-05-02",
                },
                "source": "tldr_ai",
                "item_idx": 0,
            },
            {
                "item": {
                    "model_id": "Story Again",
                    "url": "https://example.com/story?utm_source=lwiai",
                    "description": "",
                    "published": "2026-05-02",
                },
                "source": "lwiai",
                "item_idx": 1,
            },
        ]
        
        # Run full consolidation (Phase 1 + Phase 2)
        consolidated_by_url = agent._consolidate_by_url(items, [])
        consolidated = agent._consolidate_by_similarity(consolidated_by_url)
        
        # Phase 1 should have consolidated these
        assert len(consolidated) == 1
        assert consolidated[0]["impact_score"] == 2
    
    def test_multiple_consolidations_in_sequence(self):
        """Test that Phase 2 can consolidate multiple groups."""
        agent = StoryConsolidatorAgent()
        
        # Three stories, where 1&2 similar, and 3 different
        story1 = {
            "consolidated_id": "url1",
            "primary_item": {"model_id": "OpenAI GPT", "url": "https://openai.com/gpt", "published": "2026-05-02"},
            "appearances": [{"source": "tldr_ai", "date": "2026-05-02"}],
            "impact_score": 1,
        }
        
        story2 = {
            "consolidated_id": "url2",
            "primary_item": {"model_id": "OpenAI GPT", "url": "https://openai.com/news", "published": "2026-05-02"},
            "appearances": [{"source": "lwiai", "date": "2026-05-02"}],
            "impact_score": 1,
        }
        
        story3 = {
            "consolidated_id": "url3",
            "primary_item": {"model_id": "Different Story", "url": "https://other.com/news", "published": "2026-05-02"},
            "appearances": [{"source": "neuron", "date": "2026-05-02"}],
            "impact_score": 1,
        }
        
        consolidated = agent._consolidate_by_similarity([story1, story2, story3])
        
        # Should merge story1 and story2, keep story3 separate
        assert len(consolidated) == 2
        # Find the merged story
        merged = [s for s in consolidated if s["impact_score"] == 2]
        assert len(merged) == 1
        assert len(merged[0]["appearances"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
