"""
CLI entry point for llm-watch.

Usage
-----
Run a full weekly investigation report::

    python -m llmwatch.main

Or, if installed via pip::

    llm-watch

Options
-------
  --no-parallel      Disable parallel watcher execution
  --output-dir DIR   Directory to write the Markdown report (default: .)
  --dry-run          Print the report to stdout instead of writing a file
  --list-agents      List registered agents and exit
  --verbose          Enable debug logging
  --arxiv-force-fetch Bypass arXiv cache for lookup queries
  --tldr-fetch-only  Fetch TLDR items and update cache without report generation
    --vendor-blogs-fetch-only  Fetch only vendor blog feeds without report generation
    --vendor-scrape-fetch-only Fetch only vendor scrape watchers without report generation
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from typing import Any


def _parse_date_range(date_range_str: str) -> tuple[date, date] | None:
    """
    Parse a date range string in ISO 8601 format (START:END).

    Args:
        date_range_str: String like "2026-04-25:2026-05-02"

    Returns:
        (start_date, end_date) tuple if valid, None otherwise.
    """
    if not date_range_str or ":" not in date_range_str:
        return None

    parts = date_range_str.split(":")
    if len(parts) != 2:
        return None

    try:
        start_date = datetime.fromisoformat(parts[0].strip()).date()
        end_date = datetime.fromisoformat(parts[1].strip()).date()
        if start_date > end_date:
            return None
        return start_date, end_date
    except ValueError:
        return None


def _parse_agent_limit_map(spec: str | None) -> dict[str, int]:
    """
    Parse comma-separated per-agent limits.

    Expected format: "agent_name=10,other_agent=5"
    """
    if not spec:
        return {}

    parsed: dict[str, int] = {}
    chunks = [chunk.strip() for chunk in spec.split(",") if chunk.strip()]
    for chunk in chunks:
        if "=" not in chunk:
            raise ValueError(f"Invalid limit mapping '{chunk}' (expected agent=n)")

        agent_name, raw_value = chunk.split("=", 1)
        agent_name = agent_name.strip()
        raw_value = raw_value.strip()
        if not agent_name:
            raise ValueError(f"Invalid limit mapping '{chunk}' (empty agent name)")

        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid limit for '{agent_name}': '{raw_value}'") from exc

        if value < 1:
            raise ValueError(f"Invalid limit for '{agent_name}': must be >= 1")
        parsed[agent_name] = value

    return parsed


def _validate_agent_limit_map(
    limit_map: dict[str, int],
    allowed_agent_names: list[str] | tuple[str, ...],
) -> None:
    """Validate that all limit-map keys refer to known agent names."""
    allowed = set(allowed_agent_names)
    unknown = sorted(name for name in limit_map if name not in allowed)
    if not unknown:
        return

    valid = ", ".join(sorted(allowed))
    unknown_text = ", ".join(unknown)
    raise ValueError(
        f"Unknown agent name(s): {unknown_text}. Valid names: {valid}"
    )


def _resolve_agent_limit_aliases(
    limit_map: dict[str, int],
    alias_map: dict[str, str],
) -> dict[str, int]:
    """Resolve short aliases to canonical agent names."""
    resolved: dict[str, int] = {}
    for name, value in limit_map.items():
        canonical = alias_map.get(name, name)
        if canonical in resolved:
            raise ValueError(
                f"Duplicate limit for '{canonical}' via '{name}'"
            )
        resolved[canonical] = value
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-watch",
        description="Monitor LLM sources and generate a weekly investigation report.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Run watcher agents sequentially instead of in parallel threads.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory to write the Markdown report (default: current directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the report to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List all registered agents and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--arxiv-force-fetch",
        action="store_true",
        help="Bypass arXiv cache and fetch fresh results for all lookup terms.",
    )
    parser.add_argument(
        "--tldr-fetch-only",
        action="store_true",
        help="Fetch and cache TLDR watcher data only, without generating a report.",
    )
    parser.add_argument(
        "--vendor-blogs-fetch-only",
        action="store_true",
        help="Fetch only vendor blog feed watchers, without generating a report.",
    )
    parser.add_argument(
        "--vendor-scrape-fetch-only",
        action="store_true",
        help="Fetch only vendor scrape watchers, without generating a report.",
    )
    parser.add_argument(
        "--vendor-scrape-soft-fail",
        action="store_true",
        help="With --vendor-scrape-fetch-only, return exit code 0 even if one or more scrape sources fail.",
    )
    parser.add_argument(
        "--tldr-date-range",
        metavar="START:END",
        help="Fetch TLDR data for a date range (ISO 8601 format, e.g., 2026-04-25:2026-05-02). "
             "Only used with --tldr-fetch-only.",
    )
    parser.add_argument(
        "--lwiai-lookback-days",
        type=int,
        default=7,
        metavar="N",
        help="Look back N days for Last Week in AI podcast summaries (default: 7).",
    )
    parser.add_argument(
        "--neuron-lookback-days",
        type=int,
        default=7,
        metavar="N",
        help="Look back N days for The Neuron feed entries (default: 7).",
    )
    parser.add_argument(
        "--vendor-blog-lookback-days",
        type=int,
        default=14,
        metavar="N",
        help="Look back N days for vendor blog feed entries (default: 14).",
    )
    parser.add_argument(
        "--vendor-blog-max-items",
        type=int,
        default=30,
        metavar="N",
        help="Maximum items per vendor blog feed (default: 30).",
    )
    parser.add_argument(
        "--vendor-blog-feed-limits",
        metavar="AGENT=N,...",
        help=(
            "Optional per-feed item limits, e.g. "
            "openai=10,aws=5 or openai_news_feed=10,aws_ml_blog_feed=5"
        ),
    )
    parser.add_argument(
        "--vendor-scrape-lookback-days",
        type=int,
        default=14,
        metavar="N",
        help="Look back N days for vendor scrape entries (default: 14).",
    )
    parser.add_argument(
        "--vendor-scrape-max-items",
        type=int,
        default=20,
        metavar="N",
        help="Maximum items per vendor scrape source (default: 20).",
    )
    parser.add_argument(
        "--vendor-scrape-source-limits",
        metavar="AGENT=N,...",
        help=(
            "Optional per-source scrape limits, e.g. "
            "meta=10,anthropic=8 or meta_ai_blog_scrape=10"
        ),
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help=(
            "Run the Ollama editor agent after the reporter to post-process the "
            "Markdown report (adds summary, fixes truncations, annotates stale items). "
            "Requires a running Ollama server. Also enabled by LLMWATCH_EDITOR_ENABLED=true."
        ),
    )
    parser.add_argument(
        "--editor-model",
        default=None,
        metavar="MODEL",
        help=(
            "Ollama model to use for the editor pass "
            "(default: LLMWATCH_EDITOR_MODEL env var, fallback: laguna-xs.2)."
        ),
    )
    parser.add_argument(
        "--editor-skip-tasks",
        default="",
        metavar="TASKS",
        help=(
            "Comma-separated list of editor tasks to skip. "
            "Choices: summary, truncations, stale, themes, model_digest."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- Logging --------------------------------------------------------- #
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Import agents (triggers auto-registration) ---------------------- #
    from llmwatch.agents.watchers import huggingface, huggingface_papers, lastweekinai_podcast, neuron_feed, ollama, tldr_ai, vendor_blogs, vendor_scrape  # noqa: F401
    from llmwatch.agents.lookup import arxiv  # noqa: F401
    from llmwatch.agents import reporter, editor  # noqa: F401
    from llmwatch.agents.base import registry
    from llmwatch.orchestrator import Orchestrator

    if args.list_agents:
        print("Registered agents:")
        for agent in registry.agents():
            print(f"  [{agent.category:10s}]  {agent.name}")
        return 0

    if args.tldr_fetch_only:
        tldr_agent = registry.get("tldr_ai")
        if tldr_agent is None:
            logging.getLogger(__name__).error("TLDR watcher agent is not registered")
            return 1

        context = {"mode": "tldr_fetch_only"}

        # Parse and validate date range if provided
        if args.tldr_date_range:
            date_range = _parse_date_range(args.tldr_date_range)
            if date_range is None:
                logging.getLogger(__name__).error(
                    "Invalid date range format: %s. Use ISO 8601 format like '2026-04-25:2026-05-02'",
                    args.tldr_date_range,
                )
                return 1
            context["date_range"] = date_range

        result = tldr_agent.run(context=context)
        if result.errors:
            for err in result.errors:
                logging.getLogger(__name__).warning("TLDR fetch error: %s", err)
            return 1

        # Print summary based on whether date range was used
        if args.tldr_date_range:
            start_date, end_date = date_range
            print(
                f"TLDR cache updated: {len(result.data)} item(s) for date range "
                f"{start_date.isoformat()} to {end_date.isoformat()}."
            )
        else:
            today = date.today().isoformat()
            today_count = sum(1 for item in result.data if item.get("edition_date") == today)
            print(
                f"TLDR cache updated: {today_count} item(s) for {today}; "
                f"{len(result.data)} total cached item(s)."
            )
        return 0

    try:
        vendor_blog_feed_limits = _parse_agent_limit_map(args.vendor_blog_feed_limits)
        vendor_blog_feed_limits = _resolve_agent_limit_aliases(
            vendor_blog_feed_limits,
            vendor_blogs.PHASE1_VENDOR_BLOG_ALIASES,
        )
        _validate_agent_limit_map(
            vendor_blog_feed_limits,
            vendor_blogs.PHASE1_VENDOR_BLOG_AGENT_NAMES,
        )

        vendor_scrape_source_limits = _parse_agent_limit_map(args.vendor_scrape_source_limits)
        vendor_scrape_source_limits = _resolve_agent_limit_aliases(
            vendor_scrape_source_limits,
            vendor_scrape.PHASE2_VENDOR_SCRAPE_ALIASES,
        )
        _validate_agent_limit_map(
            vendor_scrape_source_limits,
            vendor_scrape.PHASE2_VENDOR_SCRAPE_AGENT_NAMES,
        )
    except ValueError as exc:
        logging.getLogger(__name__).error("Invalid vendor limit settings: %s", exc)
        return 1

    vendor_blog_context = {
        "vendor_blog_lookback_days": max(1, args.vendor_blog_lookback_days),
        "vendor_blog_max_items": max(1, args.vendor_blog_max_items),
        "vendor_blog_per_feed_max_items": vendor_blog_feed_limits,
    }

    vendor_scrape_context = {
        "vendor_scrape_lookback_days": max(1, args.vendor_scrape_lookback_days),
        "vendor_scrape_max_items": max(1, args.vendor_scrape_max_items),
        "vendor_scrape_per_source_max_items": vendor_scrape_source_limits,
    }

    if args.vendor_blogs_fetch_only:
        context = vendor_blog_context
        total_items = 0
        errors: list[str] = []

        print("Vendor blog feed fetch summary:")
        for agent_name in vendor_blogs.PHASE1_VENDOR_BLOG_AGENT_NAMES:
            agent = registry.get(agent_name)
            if agent is None:
                msg = f"{agent_name}: watcher is not registered"
                errors.append(msg)
                print(f"- {agent_name}: error ({msg})")
                continue

            result = agent.run(context=context)
            total_items += len(result.data)
            if result.errors:
                errors.extend(result.errors)
                print(f"- {agent_name}: {len(result.data)} item(s), {len(result.errors)} error(s)")
            else:
                print(f"- {agent_name}: {len(result.data)} item(s)")

        print(f"Total vendor feed items: {total_items}")
        if errors:
            for err in errors:
                logging.getLogger(__name__).warning("Vendor feed fetch error: %s", err)
            return 1
        return 0

    if args.vendor_scrape_fetch_only:
        context = vendor_scrape_context
        total_items = 0
        errors: list[str] = []
        zero_item_sources: list[tuple[str, int]] = []
        warning_threshold = vendor_scrape.get_health_warning_threshold(context)

        print("Vendor scrape fetch summary:")
        for agent_name in vendor_scrape.PHASE2_VENDOR_SCRAPE_AGENT_NAMES:
            agent = registry.get(agent_name)
            if agent is None:
                msg = f"{agent_name}: watcher is not registered"
                errors.append(msg)
                print(f"- {agent_name}: error ({msg})")
                continue

            result = agent.run(context=context)
            total_items += len(result.data)
            if len(result.data) == 0 and not result.errors:
                streak = vendor_scrape.get_health_streak(agent_name)
                if streak >= warning_threshold:
                    zero_item_sources.append((agent_name, streak))
            if result.errors:
                errors.extend(result.errors)
                print(f"- {agent_name}: {len(result.data)} item(s), {len(result.errors)} error(s)")
            else:
                print(f"- {agent_name}: {len(result.data)} item(s)")

        print(f"Total vendor scrape items: {total_items}")
        if zero_item_sources:
            print("Scrape health warnings:")
            for source_name, streak in zero_item_sources:
                print(
                    f"- {source_name}: 0 items without request errors "
                    f"(streak={streak}; possible layout drift, anti-bot behavior, or no recent posts)"
                )
        if errors:
            for err in errors:
                logging.getLogger(__name__).warning("Vendor scrape fetch error: %s", err)
            if args.vendor_scrape_soft_fail:
                print("Soft-fail enabled: returning success despite scrape fetch errors.")
                return 0
            return 1
        return 0

    output_dir = None if args.dry_run else args.output_dir

    import os as _os

    editor_enabled = args.edit or _os.getenv("LLMWATCH_EDITOR_ENABLED", "false").lower() == "true"
    editor_skip_tasks = [
        t.strip() for t in args.editor_skip_tasks.split(",") if t.strip()
    ]
    editor_options: dict[str, Any] = {"enabled": editor_enabled}
    if editor_skip_tasks:
        editor_options["skip_tasks"] = editor_skip_tasks
    if args.editor_model:
        # Override the env-var default by injecting into the process environment
        # so OllamaEditorAgent._get_config() picks it up without extra plumbing.
        _os.environ["LLMWATCH_EDITOR_MODEL"] = args.editor_model

    orchestrator = Orchestrator(
        parallel=not args.no_parallel,
        output_dir=output_dir,
        watcher_options={
            "lwiai_lookback_days": max(1, args.lwiai_lookback_days),
            "neuron_lookback_days": max(1, args.neuron_lookback_days),
            **vendor_blog_context,
            **vendor_scrape_context,
        },
        lookup_options={"arxiv_force_fetch": args.arxiv_force_fetch},
        editor_options=editor_options,
    )

    summary = orchestrator.run()

    # ---- Print report to stdout in dry-run mode -------------------------- #
    if args.dry_run:
        # Prefer the edited report when the editor ran successfully.
        editor_result = summary.get("editor_result")
        if editor_result and editor_result.data:
            for item in editor_result.data:
                report_text = item.get("report", "")
                if report_text:
                    print(report_text)
        else:
            for result in summary["reporter_results"]:
                for item in result.data:
                    report_text = item.get("report", "")
                    if report_text:
                        print(report_text)

    # ---- Summary --------------------------------------------------------- #
    watcher_count = sum(len(r.data) for r in summary["watcher_results"])
    lookup_count = sum(len(r.data) for r in summary["lookup_results"])
    errors = summary["errors"]

    logging.getLogger(__name__).info(
        "Run complete: %d models found, %d papers found, %d error(s)",
        watcher_count,
        lookup_count,
        len(errors),
    )

    if summary["report_path"]:
        print(f"\nReport written to: {summary['report_path']}")

    if errors:
        for err in errors:
            logging.getLogger(__name__).warning("Error: %s", err)

    return 1 if errors and not watcher_count and not lookup_count else 0


if __name__ == "__main__":
    sys.exit(main())
