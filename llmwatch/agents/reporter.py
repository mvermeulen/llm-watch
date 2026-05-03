"""
Reporter agent – aggregates all watcher and lookup results into a weekly
investigation report formatted as Markdown.

The reporter also scans data for previously-unseen URLs and surfaces them
as "new sources to explore" in the report.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from llmwatch.agents.base import AgentResult, BaseAgent, registry

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")


class WeeklyReporterAgent(BaseAgent):
    """
    Aggregate watcher and lookup results into a weekly Markdown report.

    Reads ``context["watcher_results"]`` and ``context["lookup_results"]``
    (both lists of :class:`~llmwatch.agents.base.AgentResult`) and produces a
    single :class:`~llmwatch.agents.base.AgentResult` whose ``data`` list
    contains one item::

        {"report": "<markdown string>", "date": "YYYY-MM-DD"}
    """

    name = "weekly_reporter"
    category = "reporter"

    def run(self, context: dict[str, Any] | None = None) -> AgentResult:
        context = context or {}
        watcher_results: list[AgentResult] = context.get("watcher_results", [])
        lookup_results: list[AgentResult] = context.get("lookup_results", [])
        consolidated_stories: list[dict[str, Any]] = context.get("consolidated_stories", [])

        today = date.today().isoformat()
        lines: list[str] = []

        lines.append(f"# LLM Watch – Weekly Investigation Report")
        lines.append(f"*Generated: {today}*")
        lines.append("")

        # ================================================================== #
        # Common Links Section (from consolidator)
        # ================================================================== #
        if consolidated_stories:
            visible_stories = [s for s in consolidated_stories if not s.get("suppressed", False)]
            suppressed_stories = [s for s in consolidated_stories if s.get("suppressed", False)]

            lines.append("## Common Links This Week")
            lines.append("")
            lines.append(
                "Repeated links surfaced across watcher feeds. Ranked by "
                "cross-source signal to highlight broadly referenced items."
            )
            lines.append("")
            
            # Rank by cross-source signal first, then impact score.
            sorted_stories = sorted(
                visible_stories,
                key=lambda s: (
                    s.get("common_link_signal", 0),
                    s.get("source_count", 0),
                    s.get("impact_score", 0),
                ),
                reverse=True,
            )

            high_signal_links = [
                s for s in sorted_stories if s.get("source_count", 0) >= 2
            ]
            repeated_references = [
                s for s in sorted_stories if s.get("source_count", 0) < 2
            ]

            def render_common_links(stories: list[dict[str, Any]], start_idx: int = 1) -> int:
                current_idx = start_idx
                for story in stories:
                    primary = story.get("primary_item", {})
                    title = primary.get("model_id", primary.get("name", "Common Link"))
                    url = primary.get("url", "")
                    desc = primary.get("description", "")
                    appearances = story.get("appearances", [])
                    impact = story.get("impact_score", 0)
                    source_count = story.get("source_count", 0)
                    link_type = story.get("common_link_type", "news_story").replace("_", " ")

                    link = f"[{title}]({url})" if url else f"**{title}**"
                    lines.append(f"### {current_idx}. {link}")
                    lines.append("")
                    current_idx += 1

                    lines.append(
                        f"**Type**: `{link_type}` | **Coverage**: {impact} reference(s) across {source_count} source(s)"
                    )
                    lines.append("")

                    if desc:
                        lines.append(desc)
                        lines.append("")

                    if appearances and len(appearances) > 1:
                        lines.append("**Seen in**:")
                        for app in appearances:
                            source = app.get("source", "unknown").replace("_", " ").title()
                            date_str = app.get("date", "")
                            date_info = f" ({date_str})" if date_str else ""
                            lines.append(f"- {source}{date_info}")
                        lines.append("")

                return current_idx

            idx = 1
            if high_signal_links:
                lines.append("### High Signal Common Links")
                lines.append("")
                idx = render_common_links(high_signal_links[:10], start_idx=idx)

            if repeated_references:
                lines.append("### Repeated References")
                lines.append("")
                render_common_links(repeated_references[:10], start_idx=idx)

            if suppressed_stories:
                lines.append("### Suppressed Repeated Links")
                lines.append("")
                lines.append(
                    "Filtered by suppression rules (for example sponsor or low-signal social links)."
                )
                lines.append("")
                reason_labels = {
                    "sponsor_link": "sponsor",
                    "single_source_social": "single-source social",
                    "domain_suppressed": "suppressed domain",
                }
                for story in suppressed_stories[:10]:
                    primary = story.get("primary_item", {})
                    title = primary.get("model_id", primary.get("name", "Suppressed Link"))
                    url = primary.get("url", "")
                    reason = story.get("suppression_reason", "suppressed")
                    reason_text = reason_labels.get(reason, reason.replace("_", " "))
                    link = f"[{title}]({url})" if url else f"**{title}**"
                    lines.append(f"- {link} ({reason_text})")
                lines.append("")
            
            lines.append("---")
            lines.append("")

        # ------------------------------------------------------------------ #
        # Section 1: Trending / new models
        # ------------------------------------------------------------------ #
        lines.append("## Trending & New Models")
        lines.append("")

        tldr_analysis_items: list[dict[str, Any]] = []
        tldr_other_items: list[dict[str, Any]] = []
        lwiai_summary_items: list[dict[str, Any]] = []
        lwiai_link_items: list[dict[str, Any]] = []
        neuron_summary_items: list[dict[str, Any]] = []
        hf_papers_items: list[dict[str, Any]] = []

        if not watcher_results:
            lines.append("*No watcher data available this week.*")
            lines.append("")
        else:
            # Sort results to show tldr_ai first, then others in registration order
            sorted_results = sorted(
                watcher_results,
                key=lambda r: (r.agent_name != "tldr_ai", watcher_results.index(r))
            )
            for w_result in sorted_results:
                if not w_result.data:
                    continue
                source_label = _source_label(w_result.agent_name)
                lines.append(f"### {source_label}")
                lines.append("")
                shown = 0
                for item in w_result.data:
                    model_id = item.get("model_id", item.get("name", "Unknown"))
                    url = item.get("url", "")
                    desc = item.get("description", "")
                    tags = item.get("tags", [])
                    source = item.get("source", "")

                    if source == "lastweekinai_podcast":
                        if "podcast_summary" in tags:
                            lwiai_summary_items.append(item)
                        elif "podcast_link" in tags:
                            lwiai_link_items.append(item)
                        continue

                    if source == "neuron":
                        neuron_summary_items.append(item)
                        continue

                    if source == "huggingface_papers":
                        hf_papers_items.append(item)
                        continue

                    # TLDR items can be routed to secondary sections.
                    if source == "tldr_ai":
                        include_in_trending = bool(item.get("include_in_trending", True))
                        local_category = str(item.get("tldr_local_category", "other"))
                        if not include_in_trending:
                            if local_category == "model_analysis":
                                tldr_analysis_items.append(item)
                            else:
                                tldr_other_items.append(item)
                            continue

                    if shown >= 15:
                        continue

                    link = f"[{model_id}]({url})" if url else f"**{model_id}**"
                    desc_str = f" – {desc}" if desc else ""
                    
                    # For TLDR items, put the single category tag at the end
                    if source == "tldr_ai" and tags:
                        category = tags[0]  # TLDR items have exactly one tag
                        lines.append(f"- {link}{desc_str} `{category}`")
                    else:
                        # For other sources, keep tags inline
                        tag_str = f" `{'` `'.join(str(t) for t in tags[:4])}`" if tags else ""
                        lines.append(f"- {link}{tag_str}{desc_str}")
                    shown += 1
                lines.append("")

        if lwiai_summary_items:
            lines.append("## Last Week in AI Podcast Summaries")
            lines.append("")
            for item in lwiai_summary_items[:5]:
                title = item.get("model_id", "Untitled episode")
                url = item.get("url", "")
                published = item.get("published", "")
                desc = item.get("description", "")
                link = f"[{title}]({url})" if url else f"**{title}**"
                date_str = f" ({published})" if published else ""
                desc_str = f" – {desc}" if desc else ""
                lines.append(f"- {link}{date_str}{desc_str}")
            lines.append("")

        if lwiai_link_items:
            lines.append("## Last Week in AI Referenced Links")
            lines.append("")
            for item in lwiai_link_items[:30]:
                title = item.get("model_id", "Referenced link")
                url = item.get("url", "")
                episode_title = item.get("episode_title", "")
                published = item.get("published", "")
                link = f"[{title}]({url})" if url else f"**{title}**"
                context_parts = [part for part in [published, episode_title] if part]
                context_str = f" ({' | '.join(context_parts)})" if context_parts else ""
                lines.append(f"- {link}{context_str}")
            lines.append("")

        if neuron_summary_items:
            lines.append("## The Neuron Summaries")
            lines.append("")
            for item in neuron_summary_items[:30]:
                title = item.get("model_id", "Untitled")
                url = item.get("url", "")
                summary = item.get("description", "")
                published = item.get("published", "")
                category = item.get("neuron_category", "")

                link = f"[{title}]({url})" if url else f"**{title}**"
                meta_parts = [part for part in [published, category] if part]
                meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""
                summary_str = f" – {summary}" if summary else ""
                lines.append(f"- {link}{meta_str}{summary_str}")
            lines.append("")

        if hf_papers_items:
            lines.append("## HuggingFace Trending Papers")
            lines.append("")
            for item in hf_papers_items[:30]:
                title = item.get("model_id", "Untitled")
                url = item.get("url", "")
                summary = item.get("description", "")
                published = item.get("published", "")
                authors = item.get("authors", "")
                upvotes = item.get("upvotes", 0)

                link = f"[{title}]({url})" if url else f"**{title}**"
                meta_parts = [part for part in [published, f"{upvotes} upvotes"] if part]
                meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""
                if authors:
                    authors_str = f"\n  *Authors: {authors}*"
                else:
                    authors_str = ""
                summary_str = f"\n  {summary}" if summary else ""
                lines.append(f"- {link}{meta_str}{authors_str}{summary_str}")
            lines.append("")

        if tldr_analysis_items:
            lines.append("## TLDR Model Analysis")
            lines.append("")
            for item in tldr_analysis_items[:20]:
                model_id = item.get("model_id", item.get("name", "Unknown"))
                url = item.get("url", "")
                desc = item.get("description", "")
                tags = item.get("tags", [])
                link = f"[{model_id}]({url})" if url else f"**{model_id}**"
                desc_str = f" – {desc}" if desc else ""
                category = tags[0] if tags else "General"
                lines.append(f"- {link}{desc_str} `{category}`")
            lines.append("")

        if tldr_other_items:
            lines.append("## TLDR Other AI News")
            lines.append("")
            for item in tldr_other_items[:20]:
                model_id = item.get("model_id", item.get("name", "Unknown"))
                url = item.get("url", "")
                desc = item.get("description", "")
                tags = item.get("tags", [])
                link = f"[{model_id}]({url})" if url else f"**{model_id}**"
                desc_str = f" – {desc}" if desc else ""
                category = tags[0] if tags else "General"
                lines.append(f"- {link}{desc_str} `{category}`")
            lines.append("")

        # ------------------------------------------------------------------ #
        # Section 2: Related research papers
        # ------------------------------------------------------------------ #
        lines.append("## Related Research Papers (arXiv)")
        lines.append("")

        all_papers: list[dict[str, Any]] = []
        for l_result in lookup_results:
            all_papers.extend(l_result.data)

        if not all_papers:
            lines.append("*No papers retrieved this week.*")
            lines.append("")
        else:
            # Group papers by query term for readability
            by_query: dict[str, list[dict]] = {}
            for paper in all_papers:
                q = paper.get("query", "general")
                by_query.setdefault(q, []).append(paper)

            for query, papers in by_query.items():
                lines.append(f"### Query: *{query}*")
                lines.append("")
                for paper in papers:
                    title = paper.get("title", "Untitled")
                    url = paper.get("url", "")
                    authors = paper.get("authors", "")
                    published = paper.get("published", "")
                    summary = paper.get("summary", "")

                    link = f"[{title}]({url})" if url else f"**{title}**"
                    meta = " | ".join(filter(None, [authors, published]))
                    lines.append(f"- {link}")
                    if meta:
                        lines.append(f"  *{meta}*")
                    if summary:
                        lines.append(f"  {summary}")
                    lines.append("")

        # ------------------------------------------------------------------ #
        # Section 3: Errors / warnings
        # ------------------------------------------------------------------ #
        all_errors = [
            e
            for result in (watcher_results + lookup_results)
            for e in result.errors
        ]
        if all_errors:
            lines.append("## Warnings")
            lines.append("")
            for err in all_errors:
                lines.append(f"- ⚠️ {err}")
            lines.append("")

        # ------------------------------------------------------------------ #
        # Section 4: New sources discovered
        # ------------------------------------------------------------------ #
        new_sources = _collect_new_sources(watcher_results + lookup_results)
        if new_sources:
            lines.append("## New Sources Discovered")
            lines.append("")
            lines.append(
                "The following URLs were found in model descriptions or paper "
                "abstracts and may be worth monitoring:"
            )
            lines.append("")
            for src in new_sources[:20]:
                lines.append(f"- {src}")
            lines.append("")

        report_text = "\n".join(lines)
        logger.info("WeeklyReporterAgent: report generated (%d chars)", len(report_text))
        return self._result(data=[{"report": report_text, "date": today}])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_label(agent_name: str) -> str:
    labels = {
        "tldr_ai": "TLDR AI – Daily Newsletter",
        "lastweekinai_podcast": "Last Week in AI – Podcast",
        "neuron_feed": "The Neuron – Feed",
        "huggingface_trending": "HuggingFace – Trending Models",
        "huggingface_trending_papers": "HuggingFace – Trending Papers",
        "ollama_models": "Ollama – Model Library",
    }
    return labels.get(agent_name, agent_name.replace("_", " ").title())


def _collect_new_sources(results: list[AgentResult]) -> list[str]:
    """Gather unique external URLs from all agent results."""
    seen: set[str] = set()
    sources: list[str] = []
    # Known hosting domains we do NOT want to surface as "new"
    known = {"huggingface.co", "ollama.com", "arxiv.org", "github.com"}

    for result in results:
        for url in result.new_sources:
            domain = _domain(url)
            if domain and domain not in known and url not in seen:
                seen.add(url)
                sources.append(url)
        # Also scan text fields in the data items
        for item in result.data:
            for field_value in item.values():
                if isinstance(field_value, str):
                    for url in _URL_RE.findall(field_value):
                        domain = _domain(url)
                        if domain and domain not in known and url not in seen:
                            seen.add(url)
                            sources.append(url)
    return sources


def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1).lstrip("www.") if m else ""


# Register a default instance in the global registry.
registry.register(WeeklyReporterAgent())
