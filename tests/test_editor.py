"""
Unit tests for the Ollama editor agent and the OllamaClient helper.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from llmwatch.agents.editor import OllamaEditorAgent, _TRUNCATION_RE
from llmwatch.agents.base import AgentResult
from llmwatch.ollama_client import OllamaClient, OllamaUnavailableError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_REPORT = """\
# LLM Watch – Weekly Investigation Report
*Generated: 2026-05-04*

## Common Links This Week

### 1. [Some Story](https://example.com/story)

**Type**: `news story` | **Coverage**: 2 reference(s)

A great story about AI.

**Seen in**:
- Tldr Ai (2026-05-01)
- Anthropic News Scrape

---

## Trending & New Models

### TLDR AI – Daily Newsletter

- [OpenAI adds animated Pets](https://openai.com/codex-pets) – OpenAI updated Codex with animated Pets and \
config imports to enhance `Headlines & Launches`
- [Speculative Decoding](https://arxiv.org/abs/1234) – Speculative decoding was applied to RL rollouts \
delivering up to 1.8x throughput gains. `Engineering & Research`

### Ollama – Model Library

- [laguna-xs.2](https://ollama.com/library/laguna-xs.2) `tools` `thinking` – Laguna XS.2 is a 33B MoE model.
- [qwen3.6](https://ollama.com/library/qwen3.6) `vision` `tools` `27b` – Qwen3.6 upgrades agentic coding.

### HuggingFace – Trending Models

- [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) `transformers` `text-generation`
- [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) `transformers` `image-text-to-text`
"""

STALE_REPORT = """\
# LLM Watch – Weekly Investigation Report
*Generated: 2026-05-04*

- [Old Item](https://example.com/old) (2026-03-01) – Very old news.
- [Recent Item](https://example.com/new) (2026-04-30) – Fresh news.
"""


def _make_client(reply: str = "mock reply") -> MagicMock:
    """Return a mock OllamaClient whose chat() returns a fixed string."""
    client = MagicMock(spec=OllamaClient)
    client.is_available.return_value = True
    client.chat.return_value = reply
    return client


# ---------------------------------------------------------------------------
# OllamaClient unit tests
# ---------------------------------------------------------------------------


class TestOllamaClientChat:
    def test_successful_chat(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "  hello world  "}
        }

        with patch("llmwatch.ollama_client.requests.post", return_value=mock_response):
            client = OllamaClient(model="test-model")
            result = client.chat("ping")

        assert result == "hello world"

    def test_connection_error_raises(self):
        import requests as _req

        with patch(
            "llmwatch.ollama_client.requests.post",
            side_effect=_req.exceptions.ConnectionError("refused"),
        ):
            client = OllamaClient(model="test-model")
            with pytest.raises(OllamaUnavailableError, match="Cannot connect"):
                client.chat("ping")

    def test_http_error_raises(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        import requests as _req

        mock_response.raise_for_status.side_effect = _req.exceptions.HTTPError("500")

        with patch("llmwatch.ollama_client.requests.post", return_value=mock_response):
            client = OllamaClient(model="test-model")
            with pytest.raises(OllamaUnavailableError, match="HTTP 500"):
                client.chat("ping")

    def test_unexpected_response_shape_raises(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "shape"}

        with patch("llmwatch.ollama_client.requests.post", return_value=mock_response):
            client = OllamaClient(model="test-model")
            with pytest.raises(OllamaUnavailableError, match="response shape"):
                client.chat("ping")

    def test_is_available_true(self):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("llmwatch.ollama_client.requests.get", return_value=mock_response):
            client = OllamaClient(model="test-model")
            assert client.is_available() is True

    def test_is_available_false_on_connection_error(self):
        import requests as _req

        with patch(
            "llmwatch.ollama_client.requests.get",
            side_effect=_req.exceptions.ConnectionError,
        ):
            client = OllamaClient(model="test-model")
            assert client.is_available() is False


# ---------------------------------------------------------------------------
# OllamaEditorAgent – disabled / unavailable paths
# ---------------------------------------------------------------------------


class TestEditorAgentDisabled:
    def test_no_markdown_returns_error(self):
        agent = OllamaEditorAgent()
        with patch("llmwatch.agents.editor.OllamaClient") as MockClient:
            MockClient.return_value.is_available.return_value = True
            result = agent.run(context={})

        assert not result.ok()
        assert "No report_markdown" in result.errors[0]

    def test_ollama_unavailable_returns_error(self):
        agent = OllamaEditorAgent()
        with patch("llmwatch.agents.editor.OllamaClient") as MockClient:
            MockClient.return_value.is_available.return_value = False
            result = agent.run(context={"report_markdown": SAMPLE_REPORT})

        assert not result.ok()
        assert "not available" in result.errors[0]


# ---------------------------------------------------------------------------
# Task: summary
# ---------------------------------------------------------------------------


class TestTaskSummary:
    def test_summary_inserted_after_generated_line(self, monkeypatch):
        monkeypatch.setenv("LLMWATCH_EDITOR_SUMMARY", "true")
        monkeypatch.setenv("LLMWATCH_EDITOR_FIX_TRUNCATIONS", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_ANNOTATE_STALE", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_THEME_TAGS", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_MODEL_DIGEST", "false")

        agent = OllamaEditorAgent()
        client = _make_client("This week AI dominated with new models.")

        result_md = agent._task_summary(SAMPLE_REPORT, "2026-05-04", client)

        assert "This week AI dominated with new models." in result_md
        # Summary should appear early, before the first section heading
        summary_pos = result_md.index("This week AI dominated")
        heading_pos = result_md.index("## Common Links")
        assert summary_pos < heading_pos

    def test_summary_task_skipped_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LLMWATCH_EDITOR_SUMMARY", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_FIX_TRUNCATIONS", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_ANNOTATE_STALE", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_THEME_TAGS", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_MODEL_DIGEST", "false")

        agent = OllamaEditorAgent()
        with patch("llmwatch.agents.editor.OllamaClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.is_available.return_value = True
            result = agent.run(context={"report_markdown": SAMPLE_REPORT})

        assert result.ok()
        # No LLM calls were made since all tasks are disabled
        mock_client.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Task: truncations
# ---------------------------------------------------------------------------


class TestTaskTruncations:
    def test_truncation_regex_matches_cut_off_lines(self):
        line = "- [Foo](https://example.com) – OpenAI updated Codex with pets and config imports to enhance `Headlines & Launches`"
        matches = _TRUNCATION_RE.findall(line)
        assert len(matches) == 1

    def test_truncation_regex_does_not_match_complete_lines(self):
        line = "- [Foo](https://example.com) – A complete sentence that ends properly."
        matches = _TRUNCATION_RE.findall(line)
        assert len(matches) == 0

    def test_task_truncations_calls_chat_for_each_match(self, monkeypatch):
        agent = OllamaEditorAgent()
        client = _make_client("A complete fixed description.")

        result_md = agent._task_truncations(SAMPLE_REPORT, "2026-05-04", client)

        # The sample report has one truncated line; chat should have been called
        assert client.chat.called
        assert "A complete fixed description." in result_md

    def test_truncation_ollama_failure_leaves_line_unchanged(self):
        agent = OllamaEditorAgent()
        client = MagicMock(spec=OllamaClient)
        client.chat.side_effect = OllamaUnavailableError("down")

        # Should not raise; original text preserved
        result_md = agent._task_truncations(SAMPLE_REPORT, "2026-05-04", client)
        assert result_md is not None


# ---------------------------------------------------------------------------
# Task: stale
# ---------------------------------------------------------------------------


class TestTaskStale:
    def test_stale_items_annotated(self):
        agent = OllamaEditorAgent()
        client = _make_client()

        result_md = agent._task_stale(STALE_REPORT, "2026-05-04", client)

        assert "<!-- stale: 2026-03-01 -->" in result_md
        assert "<!-- stale: 2026-04-30 -->" not in result_md

    def test_recent_items_not_annotated(self):
        agent = OllamaEditorAgent()
        client = _make_client()

        result_md = agent._task_stale(STALE_REPORT, "2026-05-04", client)

        # 2026-04-30 is only 4 days before report date – not stale
        lines_with_new = [l for l in result_md.splitlines() if "2026-04-30" in l]
        for line in lines_with_new:
            assert "stale" not in line

    def test_invalid_report_date_returns_unchanged(self):
        agent = OllamaEditorAgent()
        result_md = agent._task_stale(STALE_REPORT, "not-a-date", _make_client())
        assert result_md == STALE_REPORT

    def test_stale_annotation_not_duplicated(self):
        """Running the task twice should not add a second annotation."""
        agent = OllamaEditorAgent()
        once = agent._task_stale(STALE_REPORT, "2026-05-04", _make_client())
        twice = agent._task_stale(once, "2026-05-04", _make_client())
        assert twice.count("<!-- stale: 2026-03-01 -->") == 1


# ---------------------------------------------------------------------------
# Task: themes
# ---------------------------------------------------------------------------


class TestTaskThemes:
    def test_theme_comment_added_to_headings(self):
        agent = OllamaEditorAgent()
        client = _make_client("model-release")

        result_md = agent._task_themes(SAMPLE_REPORT, "2026-05-04", client)

        assert "<!-- theme: model-release -->" in result_md

    def test_theme_ollama_failure_leaves_heading_unchanged(self):
        agent = OllamaEditorAgent()
        client = MagicMock(spec=OllamaClient)
        client.chat.side_effect = OllamaUnavailableError("down")

        result_md = agent._task_themes(SAMPLE_REPORT, "2026-05-04", client)
        assert "## Common Links This Week" in result_md


# ---------------------------------------------------------------------------
# Task: model_digest
# ---------------------------------------------------------------------------


class TestTaskModelDigest:
    def test_ollama_section_replaced_with_prose(self):
        agent = OllamaEditorAgent()
        client = _make_client("This week Ollama added two notable models.")

        result_md = agent._task_model_digest(SAMPLE_REPORT, "2026-05-04", client)

        assert "This week Ollama added two notable models." in result_md
        # Raw list items should be gone
        assert "- [laguna-xs.2]" not in result_md

    def test_model_digest_failure_leaves_section_unchanged(self):
        agent = OllamaEditorAgent()
        client = MagicMock(spec=OllamaClient)
        client.chat.side_effect = OllamaUnavailableError("down")

        result_md = agent._task_model_digest(SAMPLE_REPORT, "2026-05-04", client)
        assert "- [laguna-xs.2]" in result_md


# ---------------------------------------------------------------------------
# Full run: skip_tasks
# ---------------------------------------------------------------------------


class TestSkipTasks:
    def test_skip_tasks_honoured(self, monkeypatch):
        monkeypatch.setenv("LLMWATCH_EDITOR_SUMMARY", "true")
        monkeypatch.setenv("LLMWATCH_EDITOR_FIX_TRUNCATIONS", "true")
        monkeypatch.setenv("LLMWATCH_EDITOR_ANNOTATE_STALE", "true")
        monkeypatch.setenv("LLMWATCH_EDITOR_THEME_TAGS", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_MODEL_DIGEST", "false")

        agent = OllamaEditorAgent()

        with patch("llmwatch.agents.editor.OllamaClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.is_available.return_value = True
            mock_client.chat.return_value = "fixed text"

            result = agent.run(
                context={
                    "report_markdown": SAMPLE_REPORT,
                    "report_date": "2026-05-04",
                    "skip_tasks": ["summary", "truncations", "stale"],
                }
            )

        assert result.ok()
        # All enabled tasks were skipped → no chat calls
        mock_client.chat.assert_not_called()

    def test_partial_skip(self, monkeypatch):
        monkeypatch.setenv("LLMWATCH_EDITOR_SUMMARY", "true")
        monkeypatch.setenv("LLMWATCH_EDITOR_FIX_TRUNCATIONS", "true")
        monkeypatch.setenv("LLMWATCH_EDITOR_ANNOTATE_STALE", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_THEME_TAGS", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_MODEL_DIGEST", "false")

        agent = OllamaEditorAgent()

        with patch("llmwatch.agents.editor.OllamaClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.is_available.return_value = True
            mock_client.chat.return_value = "summary text"

            result = agent.run(
                context={
                    "report_markdown": SAMPLE_REPORT,
                    "report_date": "2026-05-04",
                    "skip_tasks": ["truncations"],  # skip only truncations
                }
            )

        assert result.ok()
        # Only summary should have triggered a chat call
        assert mock_client.chat.call_count >= 1
        assert result.data[0]["report"] is not None


# ---------------------------------------------------------------------------
# Result data shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_result_contains_report_and_date(self, monkeypatch):
        monkeypatch.setenv("LLMWATCH_EDITOR_SUMMARY", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_FIX_TRUNCATIONS", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_ANNOTATE_STALE", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_THEME_TAGS", "false")
        monkeypatch.setenv("LLMWATCH_EDITOR_MODEL_DIGEST", "false")

        agent = OllamaEditorAgent()

        with patch("llmwatch.agents.editor.OllamaClient") as MockClient:
            MockClient.return_value.is_available.return_value = True
            result = agent.run(
                context={"report_markdown": SAMPLE_REPORT, "report_date": "2026-05-04"}
            )

        assert result.ok()
        assert len(result.data) == 1
        assert "report" in result.data[0]
        assert result.data[0]["date"] == "2026-05-04"
        assert result.agent_name == "ollama_editor"
        assert result.category == "editor"
