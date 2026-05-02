"""
Orchestrator for llm-watch.

The orchestrator runs agents in three phases:

1. **Watcher phase** – all registered watcher agents run in parallel (or
   sequentially if parallelism is disabled) to collect raw data from
   various sources.
2. **Lookup phase** – all registered lookup agents run, with the watcher
   results available in their context so they can derive search terms.
3. **Reporter phase** – all registered reporter agents run with both
   watcher and lookup results in their context and produce the final report.

Adding a new agent is as simple as importing its module and ensuring it
calls ``registry.register()``.  The orchestrator automatically picks it
up on the next run.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from llmwatch.agents.base import AgentResult, registry

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Run all registered agents and collect their results.

    Parameters
    ----------
    parallel:
        When ``True`` (default) watcher agents are run in parallel threads.
        Set to ``False`` for simpler, sequential execution (e.g. in tests).
    output_dir:
        Directory to write the generated Markdown report.  Defaults to the
        current working directory.  Pass ``None`` to suppress file output.
    """

    def __init__(
        self,
        parallel: bool = True,
        output_dir: str | None = ".",
        lookup_options: dict[str, Any] | None = None,
    ) -> None:
        self.parallel = parallel
        self.output_dir = output_dir
        self.lookup_options = lookup_options or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Execute all phases and return a summary dict::

            {
                "watcher_results": [...],
                "lookup_results":  [...],
                "reporter_results": [...],
                "report_path": "path/to/report.md" | None,
                "errors": [...],
            }
        """
        logger.info("Orchestrator: starting run with %d agent(s)", len(registry))

        # ---- Phase 1: watchers ------------------------------------------ #
        watcher_results = self._run_phase("watcher")

        # ---- Phase 2: lookup -------------------------------------------- #
        lookup_context: dict[str, Any] = {
            "watcher_results": watcher_results,
            "options": self.lookup_options,
        }
        lookup_results = self._run_phase("lookup", context=lookup_context)

        # ---- Phase 3: reporter ------------------------------------------ #
        reporter_context: dict[str, Any] = {
            "watcher_results": watcher_results,
            "lookup_results": lookup_results,
        }
        reporter_results = self._run_phase("reporter", context=reporter_context)

        # ---- Persist report --------------------------------------------- #
        report_path: str | None = None
        if self.output_dir is not None and reporter_results:
            report_path = self._write_report(reporter_results)

        # ---- Collect all errors ----------------------------------------- #
        all_errors = [
            e
            for r in (watcher_results + lookup_results + reporter_results)
            for e in r.errors
        ]

        return {
            "watcher_results": watcher_results,
            "lookup_results": lookup_results,
            "reporter_results": reporter_results,
            "report_path": report_path,
            "errors": all_errors,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        category: str,
        context: dict[str, Any] | None = None,
    ) -> list[AgentResult]:
        agents = registry.agents(category=category)
        if not agents:
            logger.debug("Orchestrator: no agents for category '%s'", category)
            return []

        logger.info(
            "Orchestrator: running %d %s agent(s): %s",
            len(agents),
            category,
            [a.name for a in agents],
        )

        if self.parallel and category == "watcher":
            return self._run_parallel(agents, context)
        return self._run_sequential(agents, context)

    @staticmethod
    def _run_sequential(agents, context) -> list[AgentResult]:
        results = []
        for agent in agents:
            try:
                result = agent.run(context=context)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent '%s' raised an unexpected error: %s", agent.name, exc)
                results.append(
                    AgentResult(
                        agent_name=agent.name,
                        category=agent.category,
                        errors=[str(exc)],
                    )
                )
        return results

    @staticmethod
    def _run_parallel(agents, context) -> list[AgentResult]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[AgentResult] = []
        with ThreadPoolExecutor(max_workers=min(len(agents), 8)) as pool:
            futures = {pool.submit(agent.run, context): agent for agent in agents}
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Agent '%s' raised an unexpected error: %s", agent.name, exc
                    )
                    results.append(
                        AgentResult(
                            agent_name=agent.name,
                            category=agent.category,
                            errors=[str(exc)],
                        )
                    )
        return results

    def _write_report(self, reporter_results: list[AgentResult]) -> str | None:
        for result in reporter_results:
            for item in result.data:
                report_text = item.get("report")
                report_date = item.get("date", "unknown")
                if report_text:
                    filename = f"llm_watch_report_{report_date}.md"
                    path = os.path.join(self.output_dir, filename)
                    os.makedirs(self.output_dir, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(report_text)
                    logger.info("Orchestrator: report written to %s", path)
                    return path
        return None
