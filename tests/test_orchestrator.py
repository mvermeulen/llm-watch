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
