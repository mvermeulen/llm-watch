"""
Editor agent – post-processes the generated Markdown report using a local
Ollama model.

The editor runs as Phase 4 in the orchestrator pipeline, after the reporter
has produced the Markdown string but before it is written to disk.  Each
editing task is a focused, independent prompt so the model handles a bounded
chunk rather than the entire document at once.

Tasks (all individually togglable via environment variables)
------------------------------------------------------------
- **summary**        Add a narrative executive summary after the report title.
- **truncations**    Detect and clean up descriptions that end mid-sentence.
- **stale**          Annotate items whose date is >30 days before the report date.
- **themes**         Add ``<!-- theme: … -->`` comment tags per section heading.
- **model_digest**   Collapse the Ollama/HuggingFace model tables into prose.

Configuration
-------------
All settings are read from environment variables at runtime (not at import
time) so that tests can patch ``os.environ`` without side effects.

========================  ============================  ===========
Variable                  Default                       Description
========================  ============================  ===========
LLMWATCH_EDITOR_ENABLED   false                         Master switch
LLMWATCH_EDITOR_MODEL     laguna-xs.2                   Ollama model
LLMWATCH_EDITOR_BASE_URL  http://localhost:11434         Ollama server
LLMWATCH_EDITOR_TIMEOUT   120                           Seconds per call
LLMWATCH_EDITOR_SUMMARY   true                          Enable summary task
LLMWATCH_EDITOR_FIX_TRUNCATIONS  true                   Enable truncation fix
LLMWATCH_EDITOR_ANNOTATE_STALE   true                   Enable stale annotation
LLMWATCH_EDITOR_THEME_TAGS       false                  Enable theme tags
LLMWATCH_EDITOR_MODEL_DIGEST     false                  Enable model digest
========================  ============================  ===========
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, timedelta
from typing import Any

from llmwatch.agents.base import AgentResult, BaseAgent, registry
from llmwatch.ollama_client import OllamaClient, OllamaUnavailableError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRUNCATION_RE = re.compile(
    r"(?m)^(- \[.+?\]\(.+?\) – .{20,}[a-z,])(\s*`[^`]+`\s*)$"
)
"""
Match TLDR-style list entries whose description trails off before a tag.
Group 1 = the cut-off description, group 2 = the dangling tag.
"""

_DATE_RE = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
"""Pick dates written as (YYYY-MM-DD) in item lines."""


def _get_config() -> dict[str, Any]:
    """Read all editor configuration from environment variables."""

    def _bool(name: str, default: str) -> bool:
        return os.getenv(name, default).lower() == "true"

    return {
        "enabled": _bool("LLMWATCH_EDITOR_ENABLED", "false"),
        "model": os.getenv("LLMWATCH_EDITOR_MODEL", "laguna-xs.2"),
        "base_url": os.getenv("LLMWATCH_EDITOR_BASE_URL", "http://localhost:11434"),
        "timeout": int(os.getenv("LLMWATCH_EDITOR_TIMEOUT", "120")),
        "task_summary": _bool("LLMWATCH_EDITOR_SUMMARY", "true"),
        "task_truncations": _bool("LLMWATCH_EDITOR_FIX_TRUNCATIONS", "true"),
        "task_stale": _bool("LLMWATCH_EDITOR_ANNOTATE_STALE", "true"),
        "task_themes": _bool("LLMWATCH_EDITOR_THEME_TAGS", "false"),
        "task_model_digest": _bool("LLMWATCH_EDITOR_MODEL_DIGEST", "false"),
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class OllamaEditorAgent(BaseAgent):
    """
    Post-process the weekly Markdown report using a local Ollama model.

    Input context keys
    ------------------
    report_markdown : str
        The Markdown string produced by :class:`~llmwatch.agents.reporter.WeeklyReporterAgent`.
    report_date : str
        ISO date string (YYYY-MM-DD) used for stale-item detection.
    skip_tasks : list[str]
        Optional list of task names to skip for this run, supplied by the
        orchestrator from the ``--editor-skip-tasks`` CLI flag.

    Output
    ------
    A single :class:`~llmwatch.agents.base.AgentResult` whose ``data`` list
    contains one item::

        {"report": "<edited markdown>", "date": "YYYY-MM-DD"}
    """

    name = "ollama_editor"
    category = "editor"

    TASK_NAMES = ("summary", "truncations", "stale", "themes", "model_digest")

    def run(self, context: dict[str, Any] | None = None) -> AgentResult:
        context = context or {}
        config = _get_config()

        markdown: str = context.get("report_markdown", "")
        report_date_str: str = context.get("report_date", date.today().isoformat())
        skip_tasks: list[str] = context.get("skip_tasks", [])

        if not markdown:
            logger.warning("OllamaEditorAgent: no report_markdown in context, skipping")
            return self._result(errors=["No report_markdown provided in context"])

        client = OllamaClient(
            model=config["model"],
            base_url=config["base_url"],
            timeout=config["timeout"],
        )

        if not client.is_available():
            msg = f"Ollama not available at {config['base_url']}, skipping editor"
            logger.warning("OllamaEditorAgent: %s", msg)
            return self._result(errors=[msg])

        edited = markdown
        errors: list[str] = []

        task_map = {
            "summary": (config["task_summary"], self._task_summary),
            "truncations": (config["task_truncations"], self._task_truncations),
            "stale": (config["task_stale"], self._task_stale),
            "themes": (config["task_themes"], self._task_themes),
            "model_digest": (config["task_model_digest"], self._task_model_digest),
        }

        for task_name, (enabled, fn) in task_map.items():
            if task_name in skip_tasks:
                logger.debug("OllamaEditorAgent: skipping task '%s' (explicit skip)", task_name)
                continue
            if not enabled:
                logger.debug("OllamaEditorAgent: skipping task '%s' (disabled by config)", task_name)
                continue

            logger.info("OllamaEditorAgent: running task '%s'", task_name)
            try:
                edited = fn(edited, report_date_str, client)
            except OllamaUnavailableError as exc:
                msg = f"Task '{task_name}' failed: {exc}"
                logger.error("OllamaEditorAgent: %s", msg)
                errors.append(msg)
                # Continue with remaining tasks using the last good state.
            except Exception as exc:  # noqa: BLE001
                msg = f"Task '{task_name}' raised unexpected error: {exc}"
                logger.exception("OllamaEditorAgent: %s", msg)
                errors.append(msg)

        try:
            report_date = report_date_str
        except Exception:
            report_date = date.today().isoformat()

        return self._result(
            data=[{"report": edited, "date": report_date}],
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def _task_summary(
        self, markdown: str, report_date: str, client: OllamaClient
    ) -> str:
        """Inject a narrative executive summary after the report title block."""
        # Extract the first ~3000 chars as a representative sample for the model.
        sample = markdown[:3000]

        system = (
            "You are an expert editor for a weekly AI industry newsletter. "
            "Write concisely and factually. Do not add opinions or hype."
        )
        prompt = (
            f"Below is the opening section of a weekly LLM-watch report dated {report_date}.\n\n"
            f"```\n{sample}\n```\n\n"
            "Write a 4–6 sentence executive summary paragraph that captures the dominant "
            "themes of this week's report (model releases, partnerships, product launches, "
            "research). Write plain prose, no bullet points, no heading. "
            "Output ONLY the summary paragraph, nothing else."
        )

        summary = client.chat(prompt, system=system)

        # Insert after the `*Generated: …*` line
        insert_marker = re.compile(r"(\*Generated: [^\n]+\*\n)")
        replacement = rf"\1\n{summary}\n"
        updated = insert_marker.sub(replacement, markdown, count=1)

        # Fallback: if marker not found, prepend after the first heading
        if updated == markdown:
            updated = markdown.replace("\n## ", f"\n{summary}\n\n## ", 1)

        return updated

    def _task_truncations(
        self, markdown: str, report_date: str, client: OllamaClient
    ) -> str:
        """
        Find descriptions that are cut off mid-sentence and ask the model to
        either complete them from context or trim to the last complete sentence.
        """
        truncated_lines = _TRUNCATION_RE.findall(markdown)
        if not truncated_lines:
            logger.debug("OllamaEditorAgent: no truncated descriptions found")
            return markdown

        logger.info(
            "OllamaEditorAgent: found %d potentially truncated descriptions",
            len(truncated_lines),
        )

        system = (
            "You are an editor fixing truncated text. "
            "Respond with ONLY the corrected description text, no extra commentary."
        )

        def _fix_match(match: re.Match) -> str:
            description = match.group(1)
            tag = match.group(2)
            prompt = (
                "The following news item description appears to be cut off mid-sentence. "
                "Trim it to end at the last complete sentence (keep it factual and brief). "
                "Return ONLY the fixed description text — no markdown, no tags, no extra text.\n\n"
                f"Description: {description.strip()}"
            )
            try:
                fixed = client.chat(prompt, system=system)
                return f"{fixed}{tag}"
            except OllamaUnavailableError:
                return match.group(0)  # leave unchanged on failure

        return _TRUNCATION_RE.sub(_fix_match, markdown)

    def _task_stale(
        self, markdown: str, report_date: str, client: OllamaClient
    ) -> str:
        """
        Annotate items whose date is more than 30 days before the report date
        with a ``<!-- stale: YYYY-MM-DD -->`` HTML comment inline.

        This is a purely mechanical transformation – no LLM call needed.
        The client parameter is accepted for interface consistency.
        """
        try:
            cutoff = date.fromisoformat(report_date) - timedelta(days=30)
        except ValueError:
            logger.warning("OllamaEditorAgent: invalid report_date '%s', skipping stale task", report_date)
            return markdown

        def _annotate_line(line: str) -> str:
            for m in _DATE_RE.finditer(line):
                try:
                    item_date = date.fromisoformat(m.group(1))
                except ValueError:
                    continue
                if item_date < cutoff:
                    annotation = f" <!-- stale: {m.group(1)} -->"
                    # Insert annotation at end of line if not already present
                    if "<!-- stale:" not in line:
                        return line.rstrip() + annotation
            return line

        lines = markdown.splitlines()
        return "\n".join(_annotate_line(line) for line in lines)

    def _task_themes(
        self, markdown: str, report_date: str, client: OllamaClient
    ) -> str:
        """
        Add ``<!-- theme: … -->`` comment after each major section heading.

        The model is shown the heading and the first few items in that section
        and returns a short slug (e.g. ``voice-ai``, ``agentic-coding``).
        """
        system = (
            "You are a taxonomy editor. Respond with a single short kebab-case slug "
            "(e.g. 'voice-ai', 'agentic-coding', 'model-release'). No other text."
        )

        heading_re = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)

        def _tag_heading(match: re.Match) -> str:
            heading = match.group(1)
            # Grab a small window of text after the heading for context
            start = match.end()
            snippet = markdown[start : start + 400].strip()
            prompt = (
                f"Section heading: {heading}\n\n"
                f"First items in section:\n{snippet}\n\n"
                "Return a single kebab-case theme slug for this section."
            )
            try:
                slug = client.chat(prompt, system=system).strip().lower()
                # Guard against multi-word or multi-line responses
                slug = slug.splitlines()[0].split()[0]
                return f"{heading} <!-- theme: {slug} -->"
            except OllamaUnavailableError:
                return heading

        return heading_re.sub(_tag_heading, markdown)

    def _task_model_digest(
        self, markdown: str, report_date: str, client: OllamaClient
    ) -> str:
        """
        Replace the flat Ollama and HuggingFace model list sections with a
        concise prose digest paragraph.
        """
        system = (
            "You are a technical writer summarising AI model releases. "
            "Be concise and factual. No bullet points. Plain prose only."
        )

        # Match the Ollama section
        ollama_re = re.compile(
            r"(### Ollama – Model Library\n+)((?:- \[.+\n?)+)",
            re.MULTILINE,
        )
        # Match the HuggingFace trending models section
        hf_re = re.compile(
            r"(### HuggingFace – Trending Models\n+)((?:- \[.+\n?)+)",
            re.MULTILINE,
        )

        def _digest_section(match: re.Match, section_label: str) -> str:
            heading = match.group(1)
            items_text = match.group(2).strip()
            prompt = (
                f"The following is the '{section_label}' section of a weekly AI report:\n\n"
                f"{items_text}\n\n"
                "Write a 3–5 sentence prose digest highlighting the most notable new models, "
                "their key capabilities, and any emerging patterns (e.g. tool-use focus, "
                "MoE architectures, multimodal). Output ONLY the paragraph."
            )
            try:
                digest = client.chat(prompt, system=system)
                return f"{heading}\n{digest}\n\n"
            except OllamaUnavailableError:
                return match.group(0)

        markdown = ollama_re.sub(
            lambda m: _digest_section(m, "Ollama – Model Library"), markdown
        )
        markdown = hf_re.sub(
            lambda m: _digest_section(m, "HuggingFace – Trending Models"), markdown
        )
        return markdown


# Register the agent.  The orchestrator only runs it when explicitly enabled
# via the editor phase – it is NOT placed in the "reporter" category so it
# won't run automatically alongside the consolidator/reporter.
registry.register(OllamaEditorAgent())
