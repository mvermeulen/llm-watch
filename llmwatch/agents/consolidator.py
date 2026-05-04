"""
Consolidator agent – deduplicates stories across sources.

Phase 1 MVP: URL-based deduplication only (~70% coverage) ✓
Phase 2: Add title similarity matching (~90%) ✓
Phase 3: Add Ollama semantic matching (~95%+)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

from llmwatch.agents.base import AgentResult, BaseAgent, registry

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """
    Normalize URL for comparison: remove query params, fragments, trailing slashes.
    
    Examples:
        https://example.com/article?utm_source=tldr → https://example.com/article
        https://example.com/article/ → https://example.com/article
    """
    if not url:
        return ""
    
    try:
        parsed = urlparse(url)
        # Reconstruct without query/fragment, remove trailing slash
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        return normalized
    except Exception as e:
        logger.debug("Failed to normalize URL '%s': %s", url, e)
        return ""


def _get_config() -> dict[str, Any]:
    """Load configuration from environment variables."""
    suppressed_domains = {
        d.strip().lower().lstrip("www.")
        for d in os.getenv("LLMWATCH_CONSOLIDATOR_SUPPRESS_DOMAINS", "").split(",")
        if d.strip()
    }
    allow_domains = {
        d.strip().lower().lstrip("www.")
        for d in os.getenv("LLMWATCH_CONSOLIDATOR_ALLOW_DOMAINS", "").split(",")
        if d.strip()
    }
    return {
        "similarity_threshold": float(
            os.getenv("LLMWATCH_CONSOLIDATOR_SIMILARITY_THRESHOLD", "0.85")
        ),
        "temporal_window_days": int(
            os.getenv("LLMWATCH_CONSOLIDATOR_TEMPORAL_WINDOW_DAYS", "7")
        ),
        "ollama_enabled": os.getenv("LLMWATCH_CONSOLIDATOR_OLLAMA_ENABLED", "false").lower() == "true",
        "ollama_model": os.getenv("LLMWATCH_CONSOLIDATOR_OLLAMA_MODEL", "llama3.2:3b"),
        "suppressed_domains": suppressed_domains,
        "allow_domains": allow_domains,
        "suppress_sponsors": os.getenv("LLMWATCH_CONSOLIDATOR_SUPPRESS_SPONSORS", "true").lower() == "true",
        "suppress_social_single_source": os.getenv(
            "LLMWATCH_CONSOLIDATOR_SUPPRESS_SOCIAL_SINGLE_SOURCE", "true"
        ).lower()
        == "true",
    }


class StoryConsolidatorAgent(BaseAgent):
    """
    Consolidates duplicate stories across multiple watcher and lookup sources.
    
    Phase 1 ✓: URL-based deduplication (~70% coverage)
    Phase 2 ✓: Title similarity matching within temporal window (~90% coverage)
    Phase 3 (future): Ollama semantic deduplication (~95%+)
    
    Input context:
        - watcher_results: list of AgentResult from all watchers
        - lookup_results: list of AgentResult from all lookup agents
    
    Output:
        - AgentResult with data containing consolidated stories.
          Each story includes metadata about which sources covered it.
    
    Configuration (environment variables):
        - LLMWATCH_CONSOLIDATOR_SIMILARITY_THRESHOLD: 0.85 (Phase 2)
        - LLMWATCH_CONSOLIDATOR_TEMPORAL_WINDOW_DAYS: 7 (Phase 2)
        - LLMWATCH_CONSOLIDATOR_OLLAMA_ENABLED: false (Phase 3)
        - LLMWATCH_CONSOLIDATOR_OLLAMA_MODEL: llama3.2:3b (Phase 3)
    """
    
    name = "story_consolidator"
    category = "reporter"
    
    def __init__(self):
        self.config = _get_config()

    def _source_class(self, source: str) -> str:
        """Map watcher source name to a broad source class for diversity scoring."""
        mapping = {
            "tldr_ai": "newsletter",
            "neuron_feed": "newsletter",
            "lastweekinai_podcast": "podcast",
            "lwiai": "podcast",
            "openai_news_feed": "vendor_blog",
            "google_ai_blog_feed": "vendor_blog",
            "deepmind_blog_feed": "vendor_blog",
            "microsoft_ai_blog_feed": "vendor_blog",
            "aws_ml_blog_feed": "vendor_blog",
            "qwen_blog_feed": "vendor_blog",
            "meta_ai_blog_scrape": "vendor_blog",
            "anthropic_news_scrape": "vendor_blog",
            "mistral_news_scrape": "vendor_blog",
            "xai_news_scrape": "vendor_blog",
            "huggingface_trending": "model_hub",
            "huggingface_trending_papers": "research_hub",
            "ollama_models": "model_library",
        }
        return mapping.get(source, "other")

    def _domain(self, url: str) -> str:
        if not url:
            return ""
        try:
            return urlparse(url).netloc.lower().lstrip("www.")
        except Exception:
            return ""

    def _suppression_reason(self, story: dict[str, Any]) -> str:
        """Return suppression reason, or empty string if story should be kept."""
        primary = story.get("primary_item", {})
        url = primary.get("url", "")
        domain = self._domain(url)
        link_type = story.get("common_link_type", "news_story")
        source_count = story.get("source_count", 0)

        allow_domains = self.config.get("allow_domains", set())
        suppressed_domains = self.config.get("suppressed_domains", set())

        if domain and domain in allow_domains:
            return ""

        if domain and domain in suppressed_domains:
            return "domain_suppressed"

        if self.config.get("suppress_sponsors", True) and link_type == "sponsor":
            return "sponsor_link"

        if (
            self.config.get("suppress_social_single_source", True)
            and link_type == "social_post"
            and source_count < 2
        ):
            return "single_source_social"

        return ""

    def _calculate_common_link_signal(self, appearances: list[dict[str, Any]]) -> int:
        """
        Score common links using class diversity, then source diversity, then repetition.

        This gives higher value to links seen across independent source classes
        (for example newsletter + podcast) than repeated references from one feed.
        """
        if not appearances:
            return 0

        class_weights = {
            "newsletter": 8,
            "podcast": 8,
            "vendor_blog": 7,
            "research_hub": 6,
            "model_hub": 4,
            "model_library": 4,
            "other": 5,
        }

        unique_sources = {a.get("source", "") for a in appearances if a.get("source")}
        unique_classes = {
            self._source_class(source)
            for source in unique_sources
            if source
        }

        class_diversity_score = sum(
            class_weights.get(source_class, class_weights["other"])
            for source_class in unique_classes
        )
        source_count = len(unique_sources)
        appearance_count = len(appearances)
        freshness_score, novelty_score = self._calculate_freshness_novelty_scores(
            appearances
        )

        total = (
            class_diversity_score
            + (source_count * 3)
            + appearance_count
            + freshness_score
            + novelty_score
        )
        return max(0, total)

    def _parse_appearance_date(self, date_str: str) -> datetime | None:
        """Parse appearance date (YYYY-MM-DD or ISO-8601); return None on failure."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.split("T")[0])
        except (ValueError, AttributeError, IndexError):
            return None

    def _calculate_freshness_novelty_scores(
        self,
        appearances: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """
        Compute freshness and novelty adjustments.

        Freshness rewards links seen recently and penalizes stale links.
        Novelty lightly boosts newly-emerged links and downweights evergreen links
        that recur across a wide timespan.
        """
        parsed_dates = [
            parsed
            for parsed in (
                self._parse_appearance_date(a.get("date", ""))
                for a in appearances
            )
            if parsed is not None
        ]
        if not parsed_dates:
            return (0, 0)

        now = datetime.now().date()
        latest = max(parsed_dates).date()
        earliest = min(parsed_dates).date()
        days_since_latest = (now - latest).days
        span_days = (latest - earliest).days

        # Freshness: strong reward for links still being referenced this week.
        if days_since_latest <= 1:
            freshness = 6
        elif days_since_latest <= 3:
            freshness = 4
        elif days_since_latest <= 7:
            freshness = 2
        elif days_since_latest <= 14:
            freshness = 0
        else:
            freshness = -4

        # Novelty: reward newly surfaced links, penalize long-lived evergreen links.
        novelty = 0
        if len(appearances) <= 2 and days_since_latest <= 7:
            novelty += 2
        if span_days > 14:
            novelty -= 3
        elif span_days > 7:
            novelty -= 1

        return (freshness, novelty)

    def _classify_common_link_type(self, item: dict[str, Any]) -> str:
        """
        Classify common links using URL/domain/path and simple text heuristics.

        Returns one of:
        - sponsor
        - paper
        - repo
        - social_post
        - model_card
        - news_story
        """
        title = (item.get("model_id") or item.get("name") or "").lower()
        description = (item.get("description") or "").lower()
        text = f"{title} {description}"

        if "sponsor" in text or "sponsored" in text:
            return "sponsor"

        url = item.get("url", "")
        if not url:
            return "news_story"

        parsed = urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
        path = parsed.path.lower()

        if domain == "arxiv.org" or path.startswith("/abs/") or path.startswith("/pdf/"):
            return "paper"

        if domain in {"github.com", "gitlab.com", "bitbucket.org"}:
            return "repo"

        if domain in {
            "x.com",
            "twitter.com",
            "threadreaderapp.com",
            "linkedin.com",
            "reddit.com",
            "youtube.com",
            "tiktok.com",
            "news.ycombinator.com",
        }:
            return "social_post"

        if domain == "huggingface.co":
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2 and parts[0] != "papers":
                return "model_card"
            if path.startswith("/papers"):
                return "paper"

        if domain == "ollama.com" and path.startswith("/library/"):
            return "model_card"

        if any(
            key in path
            for key in (
                "/news",
                "/blog",
                "/article",
                "/articles",
                "/newsletter",
                "/explainer-articles",
                "/story",
            )
        ):
            return "news_story"

        return "news_story"
    
    def run(self, context: dict[str, Any] | None = None) -> AgentResult:
        context = context or {}
        watcher_results = context.get("watcher_results", [])
        lookup_results = context.get("lookup_results", [])
        
        # Collect all items from all watchers with source tracking
        all_items_with_source = []
        for w_result in watcher_results:
            for idx, item in enumerate(w_result.data):
                all_items_with_source.append({
                    "item": item,
                    "source": w_result.agent_name,
                    "item_idx": idx,
                })
        
        # Phase 1: Consolidate by URL
        consolidated = self._consolidate_by_url(all_items_with_source, lookup_results)
        
        # Phase 2: Consolidate by title similarity (for no-URL and mismatched stories)
        consolidated = self._consolidate_by_similarity(consolidated)
        
        logger.info(
            "StoryConsolidatorAgent: consolidated %d items into %d unique stories",
            len(all_items_with_source),
            len(consolidated),
        )
        
        return self._result(data=consolidated)
    
    def _consolidate_by_url(
        self,
        items_with_source: list[dict[str, Any]],
        lookup_results: list[AgentResult],
    ) -> list[dict[str, Any]]:
        """
        Group items by normalized URL. Items with same URL become one consolidated story.
        
        Returns list of consolidated story dicts with metadata.
        """
        # Map: normalized_url → list of (item, source, item_idx)
        url_groups: dict[str, list[dict]] = {}
        url_to_primary: dict[str, dict] = {}
        
        for item_with_source in items_with_source:
            item = item_with_source["item"]
            source = item_with_source["source"]
            url = item.get("url", "")
            normalized = _normalize_url(url)
            
            if not normalized:
                # Items without URLs are treated individually (remain common links)
                consolidated_id = f"no_url_{len(url_groups)}"
                if consolidated_id not in url_groups:
                    url_groups[consolidated_id] = []
                    url_to_primary[consolidated_id] = item
                url_groups[consolidated_id].append({
                    "item": item,
                    "source": source,
                    "item_idx": item_with_source["item_idx"],
                })
                continue
            
            # Group by normalized URL
            if normalized not in url_groups:
                url_groups[normalized] = []
                url_to_primary[normalized] = item
            
            url_groups[normalized].append({
                "item": item,
                "source": source,
                "item_idx": item_with_source["item_idx"],
            })
        
        # Build consolidated stories
        consolidated = []
        for url_key, items_group in url_groups.items():
            # Pick the most detailed item as primary
            primary = max(
                [item_info["item"] for item_info in items_group],
                key=lambda x: len(x.get("description", "")),
            )
            
            appearances = []
            for item_info in items_group:
                item = item_info["item"]
                appearance = {
                    "source": item_info["source"],
                    "date": item.get("published", ""),
                    "url": item.get("url", ""),
                    "title": item.get("model_id", item.get("name", "")),
                }

                # Add source-specific metadata
                if "episode_title" in item:
                    appearance["episode"] = item["episode_title"]
                if "neuron_category" in item:
                    appearance["category"] = item["neuron_category"]
                if "tags" in item and item["tags"]:
                    appearance["tags"] = item["tags"]

                appearances.append(appearance)

            unique_sources = {
                appearance.get("source", "")
                for appearance in appearances
                if appearance.get("source")
            }
            source_count = len(unique_sources)

            story = {
                "consolidated_id": url_key,
                "primary_item": primary,
                "appearances": appearances,
                "impact_score": len(appearances),
                "source_count": source_count,
                "item_type": "common_link",
                "common_link_type": self._classify_common_link_type(primary),
                "freshness_score": self._calculate_freshness_novelty_scores(appearances)[0],
                "novelty_score": self._calculate_freshness_novelty_scores(appearances)[1],
                "common_link_signal": self._calculate_common_link_signal(appearances),
            }
            suppression_reason = self._suppression_reason(story)
            story["suppressed"] = bool(suppression_reason)
            story["suppression_reason"] = suppression_reason
            
            consolidated.append(story)
        
        return consolidated
    
    def _extract_title(self, item: dict[str, Any]) -> str:
        """Extract title from item for similarity matching."""
        return (item.get("model_id") or item.get("name") or "").strip().lower()
    
    def _calculate_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles using SequenceMatcher.
        Returns a score between 0 and 1, where 1 is identical.
        Case-insensitive comparison.
        """
        if not title1 or not title2:
            return 0.0
        # Normalize to lowercase for case-insensitive comparison
        t1 = title1.strip().lower()
        t2 = title2.strip().lower()
        matcher = SequenceMatcher(None, t1, t2)
        return matcher.ratio()
    
    def _consolidate_by_similarity(self, consolidated: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Phase 2: Further consolidate stories by title similarity for items without good URL matches.
        
        This handles:
        - Items without URLs that describe the same story
        - Different URLs with very similar titles (e.g., typos)
        - Cross-source coverage within temporal window
        
        Args:
            consolidated: List of consolidated stories from Phase 1 (by URL)
            
        Returns:
            Updated list with additional consolidation by similarity
        """
        if len(consolidated) < 2:
            return consolidated
        
        similarity_threshold = self.config["similarity_threshold"]
        temporal_window_days = self.config["temporal_window_days"]
        
        # Extract reference date from most recent appearance
        now = datetime.now()
        cutoff_date = now - timedelta(days=temporal_window_days)
        
        def is_within_window(date_str: str) -> bool:
            """Check if date string is within temporal window."""
            try:
                # Try parsing as ISO format (YYYY-MM-DD)
                item_date = datetime.fromisoformat(date_str.split("T")[0])
                return item_date >= cutoff_date.replace(hour=0, minute=0, second=0, microsecond=0)
            except (ValueError, AttributeError, IndexError):
                # If parsing fails, include it (assume it's recent)
                return True
        
        # Build a mapping of stories to merge
        merged_indices = set()
        merges = []  # List of (idx1, idx2) pairs to merge
        
        for i in range(len(consolidated)):
            if i in merged_indices:
                continue
                
            story_i = consolidated[i]
            primary_i = story_i["primary_item"]
            title_i = self._extract_title(primary_i)
            
            # Skip if title is empty (e.g., only no-URL items without names)
            if not title_i:
                continue
            
            for j in range(i + 1, len(consolidated)):
                if j in merged_indices:
                    continue
                    
                story_j = consolidated[j]
                primary_j = story_j["primary_item"]
                title_j = self._extract_title(primary_j)
                
                if not title_j:
                    continue
                
                # Check temporal overlap - both stories must have at least one date within window
                dates_i = [app.get("date", "") for app in story_i.get("appearances", [])]
                dates_j = [app.get("date", "") for app in story_j.get("appearances", [])]
                
                # Find if each story has at least one date within the window
                has_recent_date_i = any(is_within_window(date) for date in dates_i)
                has_recent_date_j = any(is_within_window(date) for date in dates_j)
                
                # Both stories must have at least one recent date to consolidate
                if not (has_recent_date_i and has_recent_date_j):
                    continue
                
                # Check title similarity
                similarity = self._calculate_similarity(title_i, title_j)
                
                if similarity >= similarity_threshold:
                    merges.append((i, j))
                    merged_indices.add(j)
                    logger.debug(
                        "Merging stories: '%s' ≈ '%s' (similarity=%.2f)",
                        title_i,
                        title_j,
                        similarity,
                    )
        
        # Apply merges
        result = []
        merged_stories = {}  # Map of merged story indices to merge targets
        
        for i, j in merges:
            merged_stories[j] = i
        
        for i, story in enumerate(consolidated):
            if i in merged_stories:
                # Merge this story into another
                target_idx = merged_stories[i]
                # We'll handle this by updating the target later
                continue
            
            # Check if any stories should be merged into this one
            stories_to_merge = [consolidated[j] for i_orig, j in merges if i_orig == i]
            
            if stories_to_merge:
                # Merge additional stories into this one
                for story_to_merge in stories_to_merge:
                    # Add appearances from merged story
                    story["appearances"].extend(story_to_merge.get("appearances", []))
                    # Update impact score
                    story["impact_score"] = len(story["appearances"])
                    story["source_count"] = len(
                        {
                            app.get("source", "")
                            for app in story["appearances"]
                            if app.get("source")
                        }
                    )
                    story["common_link_signal"] = self._calculate_common_link_signal(
                        story["appearances"]
                    )
                    freshness_score, novelty_score = self._calculate_freshness_novelty_scores(
                        story["appearances"]
                    )
                    story["freshness_score"] = freshness_score
                    story["novelty_score"] = novelty_score
                    suppression_reason = self._suppression_reason(story)
                    story["suppressed"] = bool(suppression_reason)
                    story["suppression_reason"] = suppression_reason
                    
                    logger.debug(
                        "Consolidated: '%s' now has %d appearances",
                        self._extract_title(story["primary_item"]),
                        story["impact_score"],
                    )
            
            result.append(story)
        
        return result


# Register in the global agent registry
registry.register(StoryConsolidatorAgent())
