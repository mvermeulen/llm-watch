"""
Unit tests for the Orchestrator.
"""

from unittest.mock import MagicMock, patch

import pytest

from llmwatch.agents.base import AgentRegistry, AgentResult, BaseAgent
from llmwatch.orchestrator import Orchestrator


class _FakeWatcher(BaseAgent):
    name = "fake_watcher"
    category = "watcher"

    def run(self, context=None):
        return self._result(
            data=[{"model_id": "org/TestModel", "url": "https://example.com", "source": "test"}]
        )


class _FakeLookup(BaseAgent):
    name = "fake_lookup"
    category = "lookup"

    def run(self, context=None):
        return self._result(
            data=[
                {
                    "title": "A Paper",
                    "url": "https://arxiv.org/abs/0000.00001",
                    "authors": "Foo Bar",
                    "published": "2024-01-01",
                    "summary": "Summary.",
                    "query": "TestModel",
                }
            ]
        )


class _FakeReporter(BaseAgent):
    name = "fake_reporter"
    category = "reporter"

    def run(self, context=None):
        return self._result(
            data=[{"report": "# Report\nContent here.", "date": "2024-01-01"}]
        )


class _BrokenWatcher(BaseAgent):
    name = "broken_watcher"
    category = "watcher"

    def run(self, context=None):
        raise RuntimeError("watcher exploded")


def _make_registry(*agents):
    reg = AgentRegistry()
    for a in agents:
        reg.register(a)
    return reg


class TestOrchestrator:
    def test_full_run_returns_summary(self, tmp_path):
        reg = _make_registry(_FakeWatcher(), _FakeLookup(), _FakeReporter())
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(parallel=False, output_dir=str(tmp_path))
            summary = orch.run()

        assert len(summary["watcher_results"]) == 1
        assert len(summary["lookup_results"]) == 1
        assert len(summary["reporter_results"]) == 1
        assert summary["errors"] == []

    def test_report_written_to_file(self, tmp_path):
        reg = _make_registry(_FakeWatcher(), _FakeLookup(), _FakeReporter())
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(parallel=False, output_dir=str(tmp_path))
            summary = orch.run()

        assert summary["report_path"] is not None
        report_file = summary["report_path"]
        assert report_file.endswith(".md")
        with open(report_file) as fh:
            content = fh.read()
        assert "# Report" in content

    def test_no_file_written_when_output_dir_none(self, tmp_path):
        reg = _make_registry(_FakeWatcher(), _FakeLookup(), _FakeReporter())
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(parallel=False, output_dir=None)
            summary = orch.run()

        assert summary["report_path"] is None

    def test_broken_watcher_error_captured(self, tmp_path):
        reg = _make_registry(_BrokenWatcher(), _FakeLookup(), _FakeReporter())
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(parallel=False, output_dir=str(tmp_path))
            summary = orch.run()

        # Error should be captured but not crash the run
        assert len(summary["errors"]) >= 1
        assert any("watcher exploded" in e for e in summary["errors"])

    def test_lookup_receives_watcher_context(self, tmp_path):
        """The lookup agent should receive watcher results in its context."""
        received_context = {}

        class _CaptureLookup(BaseAgent):
            name = "capture_lookup"
            category = "lookup"

            def run(self, context=None):
                received_context.update(context or {})
                return self._result()

        reg = _make_registry(_FakeWatcher(), _CaptureLookup(), _FakeReporter())
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(parallel=False, output_dir=None)
            orch.run()

        assert "watcher_results" in received_context
        assert len(received_context["watcher_results"]) == 1

    def test_watcher_receives_watcher_options_context(self, tmp_path):
        """Watcher agents should receive watcher options as run context."""
        received_context = {}

        class _CaptureWatcher(BaseAgent):
            name = "capture_watcher"
            category = "watcher"

            def run(self, context=None):
                received_context.update(context or {})
                return self._result()

        reg = _make_registry(_CaptureWatcher(), _FakeLookup(), _FakeReporter())
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(
                parallel=False,
                output_dir=None,
                watcher_options={"lwiai_lookback_days": 14},
            )
            orch.run()

        assert received_context.get("lwiai_lookback_days") == 14

    def test_reporter_receives_full_context(self, tmp_path):
        """The reporter should receive both watcher and lookup results."""
        received_context = {}

        class _CaptureReporter(BaseAgent):
            name = "capture_reporter"
            category = "reporter"

            def run(self, context=None):
                received_context.update(context or {})
                return self._result(data=[{"report": "x", "date": "2024-01-01"}])

        reg = _make_registry(_FakeWatcher(), _FakeLookup(), _CaptureReporter())
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(parallel=False, output_dir=None)
            orch.run()

        assert "watcher_results" in received_context
        assert "lookup_results" in received_context

    def test_empty_registry_returns_empty_summary(self, tmp_path):
        reg = _make_registry()
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(parallel=False, output_dir=None)
            summary = orch.run()

        assert summary["watcher_results"] == []
        assert summary["lookup_results"] == []
        assert summary["reporter_results"] == []
        assert summary["report_path"] is None
        assert summary["errors"] == []

    def test_editor_result_key_always_present(self, tmp_path):
        """Summary dict always contains 'editor_result', even when editor is disabled."""
        reg = _make_registry(_FakeWatcher(), _FakeLookup(), _FakeReporter())
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(parallel=False, output_dir=None)
            summary = orch.run()

        assert "editor_result" in summary
        assert summary["editor_result"] is None

    def test_editor_disabled_writes_reporter_output(self, tmp_path):
        """When editor is disabled the unedited reporter Markdown is written."""
        reg = _make_registry(_FakeWatcher(), _FakeLookup(), _FakeReporter())
        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(
                parallel=False,
                output_dir=str(tmp_path),
                editor_options={"enabled": False},
            )
            summary = orch.run()

        assert summary["editor_result"] is None
        assert summary["report_path"] is not None
        content = open(summary["report_path"]).read()
        assert "# Report" in content

    def test_editor_enabled_writes_edited_output(self, tmp_path):
        """When the editor runs successfully its Markdown is written, not the reporter's."""
        # Pre-import so the `import llmwatch.agents.editor` inside
        # _run_editor_phase is a no-op (module already in sys.modules).
        import llmwatch.agents.editor  # noqa: F401

        reg = _make_registry(_FakeWatcher(), _FakeLookup(), _FakeReporter())

        class _FakeEditor(BaseAgent):
            name = "ollama_editor"
            category = "editor"

            def run(self, context=None):
                return self._result(
                    data=[{"report": "# Edited Report\nPolished content.", "date": "2024-01-01"}]
                )

        reg.register(_FakeEditor())

        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(
                parallel=False,
                output_dir=str(tmp_path),
                editor_options={"enabled": True},
            )
            summary = orch.run()

        assert summary["editor_result"] is not None
        assert summary["report_path"] is not None
        content = open(summary["report_path"]).read()
        assert "Edited Report" in content

    def test_editor_failure_falls_back_to_reporter_output(self, tmp_path):
        """If the editor errors, the unedited reporter output is still written."""
        import llmwatch.agents.editor  # noqa: F401

        reg = _make_registry(_FakeWatcher(), _FakeLookup(), _FakeReporter())

        class _BrokenEditor(BaseAgent):
            name = "ollama_editor"
            category = "editor"

            def run(self, context=None):
                return self._result(errors=["Ollama not available at http://localhost:11434"])

        reg.register(_BrokenEditor())

        with patch("llmwatch.orchestrator.registry", reg):
            orch = Orchestrator(
                parallel=False,
                output_dir=str(tmp_path),
                editor_options={"enabled": True},
            )
            summary = orch.run()

        # Editor result is present but carries an error
        assert summary["editor_result"] is not None
        assert summary["editor_result"].errors
        # Unedited reporter output should still have been written
        assert summary["report_path"] is not None
        content = open(summary["report_path"]).read()
        assert "# Report" in content

    def test_editor_env_var_enables_editor(self, tmp_path, monkeypatch):
        """LLMWATCH_EDITOR_ENABLED=true triggers the editor without editor_options."""
        monkeypatch.setenv("LLMWATCH_EDITOR_ENABLED", "true")
        orch = Orchestrator(parallel=False, output_dir=None)
        assert orch.editor_options["enabled"] is True

    def test_editor_explicit_option_overrides_env_var(self, tmp_path, monkeypatch):
        """Passing editor_options={'enabled': False} wins even when env var is true."""
        monkeypatch.setenv("LLMWATCH_EDITOR_ENABLED", "true")
        orch = Orchestrator(
            parallel=False,
            output_dir=None,
            editor_options={"enabled": False},
        )
        assert orch.editor_options["enabled"] is False
