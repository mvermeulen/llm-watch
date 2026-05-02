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
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta


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
        "--tldr-date-range",
        metavar="START:END",
        help="Fetch TLDR data for a date range (ISO 8601 format, e.g., 2026-04-25:2026-05-02). "
             "Only used with --tldr-fetch-only.",
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
    from llmwatch.agents.watchers import huggingface, ollama, tldr_ai  # noqa: F401
    from llmwatch.agents.lookup import arxiv  # noqa: F401
    from llmwatch.agents import reporter  # noqa: F401
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

    output_dir = None if args.dry_run else args.output_dir

    orchestrator = Orchestrator(
        parallel=not args.no_parallel,
        output_dir=output_dir,
        lookup_options={"arxiv_force_fetch": args.arxiv_force_fetch},
    )

    summary = orchestrator.run()

    # ---- Print report to stdout in dry-run mode -------------------------- #
    if args.dry_run:
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
