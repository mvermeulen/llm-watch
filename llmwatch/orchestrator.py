"""
Orchestrator for llm-watch.

The orchestrator runs agents in four phases:

1. **Watcher phase** – all registered watcher agents run in parallel (or
   sequentially if parallelism is disabled) to collect raw data from
   various sources.
2. **Lookup phase** – all registered lookup agents run, with the watcher
   results available in their context so they can derive search terms.
3. **Reporter phase** – all registered reporter agents run with both
   watcher and lookup results in their context and produce the final report.
4. **Editor phase** – the optional Ollama editor agent post-processes the
   generated Markdown (summary injection, truncation cleanup, stale
   annotation, theme tags, model digest).  Enabled by passing
   ``editor_options`` with ``{"enabled": True}`` or by setting the
   ``LLMWATCH_EDITOR_ENABLED`` environment variable.

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
        watcher_options: dict[str, Any] | None = None,
        lookup_options: dict[str, Any] | None = None,
        editor_options: dict[str, Any] | None = None,
    ) -> None:
        self.parallel = parallel
        self.output_dir = output_dir
        self.watcher_options = watcher_options or {}
        self.lookup_options = lookup_options or {}
        opts = editor_options or {}
        # If the caller did not set "enabled", fall back to the env var so that
        # programmatic Orchestrator(...) users can enable the editor via
        # LLMWATCH_EDITOR_ENABLED without going through the CLI.
        if "enabled" not in opts:
            opts = dict(opts)  # don't mutate the caller's dict
            opts["enabled"] = os.getenv("LLMWATCH_EDITOR_ENABLED", "false").lower() == "true"
        self.editor_options = opts

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
                "editor_result": AgentResult | None,
                "report_path": "path/to/report.md" | None,
                "errors": [...],
            }
        """
        logger.info("Orchestrator: starting run with %d agent(s)", len(registry))

        # ---- Phase 1: watchers ------------------------------------------ #
        watcher_results = self._run_phase("watcher", context=self.watcher_options)

        # ---- Phase 2: lookup -------------------------------------------- #
        lookup_context: dict[str, Any] = {
            "watcher_results": watcher_results,
            "options": self.lookup_options,
        }
        lookup_results = self._run_phase("lookup", context=lookup_context)

        # ---- Phase 3: reporter ------------------------------------------ #
        # Run reporter agents sequentially to allow consolidator to enrich context
        reporter_context: dict[str, Any] = {
            "watcher_results": watcher_results,
            "lookup_results": lookup_results,
            "consolidated_stories": None,  # Will be populated by consolidator
        }
        reporter_results = self._run_reporter_phase(reporter_context)

        # ---- Phase 4: editor (optional) --------------------------------- #
        editor_result: AgentResult | None = None
        if self.editor_options.get("enabled", False):
            editor_result = self._run_editor_phase(reporter_results)

        # ---- Persist report --------------------------------------------- #
        report_path: str | None = None
        if self.output_dir is not None:
            if editor_result is not None and editor_result.data:
                report_path = self._write_report([editor_result])
            elif reporter_results:
                report_path = self._write_report(reporter_results)

        # ---- Collect all errors ----------------------------------------- #
        editor_errors = editor_result.errors if editor_result else []
        all_errors = [
            e
            for r in (watcher_results + lookup_results + reporter_results)
            for e in r.errors
        ] + editor_errors

        return {
            "watcher_results": watcher_results,
            "lookup_results": lookup_results,
            "reporter_results": reporter_results,
            "editor_result": editor_result,
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

    def _run_reporter_phase(self, context: dict[str, Any]) -> list[AgentResult]:
        """
        Run reporter agents sequentially, allowing consolidator to enrich context.
        
        The consolidator (story_consolidator agent) runs first and populates
        consolidated_stories in the context for subsequent reporter agents to use.
        """
        agents = registry.agents(category="reporter")
        if not agents:
            logger.debug("Orchestrator: no reporter agents found")
            return []

        logger.info(
            "Orchestrator: running %d reporter agent(s): %s",
            len(agents),
            [a.name for a in agents],
        )

        results = []
        for agent in agents:
            try:
                logger.debug("Running reporter agent: %s", agent.name)
                result = agent.run(context=context)
                results.append(result)
                
                # If this is the consolidator, add its output to context for next agents
                if agent.name == "story_consolidator":
                    context["consolidated_stories"] = result.data
                    logger.info(
                        "Consolidator produced %d consolidated stories",
                        len(result.data) if result.data else 0,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Reporter agent '%s' raised an unexpected error: %s",
                    agent.name,
                    exc,
                )
                results.append(
                    AgentResult(
                        agent_name=agent.name,
                        category="reporter",
                        errors=[str(exc)],
                    )
                )
        return results

    def _run_editor_phase(
        self, reporter_results: list[AgentResult]
    ) -> AgentResult | None:
        """
        Run the Ollama editor agent against the Markdown produced by the reporter.

        Returns the editor's :class:`AgentResult` (which carries the edited
        Markdown in ``data[0]["report"]``), or ``None`` if no editor agent is
        registered or the reporter produced no Markdown.
        """
        # The editor module is always imported at startup via main.py, but in
        # programmatic / test usage it may not be.  Guard with a no-op import.
        import llmwatch.agents.editor  # noqa: F401 – ensures OllamaEditorAgent is registered

        agents = registry.agents(category="editor")
        if not agents:
            logger.debug("Orchestrator: no editor agents registered")
            return None

        # Extract Markdown from the first reporter result that carries one.
        report_markdown: str = ""
        report_date: str = ""
        for result in reporter_results:
            for item in result.data:
                if "report" in item:
                    report_markdown = item["report"]
                    report_date = item.get("date", "")
                    break
            if report_markdown:
                break

        if not report_markdown:
            logger.warning("Orchestrator: editor phase skipped – no report Markdown found")
            return None

        editor_context: dict[str, Any] = {
            "report_markdown": report_markdown,
            "report_date": report_date,
            "skip_tasks": self.editor_options.get("skip_tasks", []),
        }

        logger.info("Orchestrator: running editor agent(s): %s", [a.name for a in agents])
        # Only the first registered editor agent is used.
        agent = agents[0]
        try:
            return agent.run(context=editor_context)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Editor agent '%s' raised an unexpected error: %s", agent.name, exc)
            from llmwatch.agents.base import AgentResult as _AR

            return _AR(agent_name=agent.name, category="editor", errors=[str(exc)])

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
