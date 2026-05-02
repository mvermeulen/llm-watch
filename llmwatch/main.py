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
"""

from __future__ import annotations

import argparse
import logging
import sys


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
    from llmwatch.agents.watchers import huggingface, ollama  # noqa: F401
    from llmwatch.agents.lookup import arxiv  # noqa: F401
    from llmwatch.agents import reporter  # noqa: F401
    from llmwatch.agents.base import registry
    from llmwatch.orchestrator import Orchestrator

    if args.list_agents:
        print("Registered agents:")
        for agent in registry.agents():
            print(f"  [{agent.category:10s}]  {agent.name}")
        return 0

    output_dir = None if args.dry_run else args.output_dir

    orchestrator = Orchestrator(
        parallel=not args.no_parallel,
        output_dir=output_dir,
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
