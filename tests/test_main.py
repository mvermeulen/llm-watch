"""
Unit tests for CLI argument parsing in llmwatch.main.
"""

from datetime import date
from llmwatch.main import (
    _build_parser,
    main as cli_main,
    _parse_agent_limit_map,
    _parse_date_range,
    _resolve_agent_limit_aliases,
    _validate_agent_limit_map,
)


class TestCliParser:
    def test_parses_tldr_fetch_only_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--tldr-fetch-only"])
        assert args.tldr_fetch_only is True

    def test_parses_vendor_blogs_fetch_only_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--vendor-blogs-fetch-only"])
        assert args.vendor_blogs_fetch_only is True

    def test_tldr_fetch_only_defaults_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.tldr_fetch_only is False

    def test_vendor_blogs_fetch_only_defaults_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.vendor_blogs_fetch_only is False

    def test_parses_vendor_scrape_fetch_only_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--vendor-scrape-fetch-only"])
        assert args.vendor_scrape_fetch_only is True

    def test_vendor_scrape_fetch_only_defaults_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.vendor_scrape_fetch_only is False

    def test_parses_vendor_scrape_soft_fail_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--vendor-scrape-soft-fail"])
        assert args.vendor_scrape_soft_fail is True

    def test_vendor_scrape_soft_fail_defaults_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.vendor_scrape_soft_fail is False

    def test_parses_tldr_date_range(self):
        parser = _build_parser()
        args = parser.parse_args(["--tldr-date-range", "2026-04-25:2026-05-02"])
        assert args.tldr_date_range == "2026-04-25:2026-05-02"

    def test_tldr_date_range_defaults_none(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.tldr_date_range is None

    def test_parses_lwiai_lookback_days(self):
        parser = _build_parser()
        args = parser.parse_args(["--lwiai-lookback-days", "14"])
        assert args.lwiai_lookback_days == 14

    def test_lwiai_lookback_days_defaults_to_7(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.lwiai_lookback_days == 7

    def test_parses_neuron_lookback_days(self):
        parser = _build_parser()
        args = parser.parse_args(["--neuron-lookback-days", "10"])
        assert args.neuron_lookback_days == 10

    def test_neuron_lookback_days_defaults_to_7(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.neuron_lookback_days == 7

    def test_parses_vendor_blog_lookback_days(self):
        parser = _build_parser()
        args = parser.parse_args(["--vendor-blog-lookback-days", "21"])
        assert args.vendor_blog_lookback_days == 21

    def test_vendor_blog_lookback_days_defaults_to_14(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.vendor_blog_lookback_days == 14

    def test_parses_vendor_blog_max_items(self):
        parser = _build_parser()
        args = parser.parse_args(["--vendor-blog-max-items", "12"])
        assert args.vendor_blog_max_items == 12

    def test_vendor_blog_max_items_defaults_to_30(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.vendor_blog_max_items == 30

    def test_parses_vendor_blog_feed_limits(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vendor-blog-feed-limits",
            "openai_news_feed=5,aws_ml_blog_feed=2",
        ])
        assert args.vendor_blog_feed_limits == "openai_news_feed=5,aws_ml_blog_feed=2"

    def test_parses_vendor_scrape_lookback_days(self):
        parser = _build_parser()
        args = parser.parse_args(["--vendor-scrape-lookback-days", "21"])
        assert args.vendor_scrape_lookback_days == 21

    def test_vendor_scrape_lookback_days_defaults_to_14(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.vendor_scrape_lookback_days == 14

    def test_parses_vendor_scrape_max_items(self):
        parser = _build_parser()
        args = parser.parse_args(["--vendor-scrape-max-items", "11"])
        assert args.vendor_scrape_max_items == 11

    def test_vendor_scrape_max_items_defaults_to_20(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.vendor_scrape_max_items == 20

    def test_parses_vendor_scrape_source_limits(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vendor-scrape-source-limits",
            "meta=4,anthropic=2",
        ])
        assert args.vendor_scrape_source_limits == "meta=4,anthropic=2"


class TestDateRangeParser:
    def test_parse_valid_date_range(self):
        result = _parse_date_range("2026-04-25:2026-05-02")
        assert result is not None
        start, end = result
        assert start == date(2026, 4, 25)
        assert end == date(2026, 5, 2)

    def test_parse_single_day_range(self):
        result = _parse_date_range("2026-05-01:2026-05-01")
        assert result is not None
        start, end = result
        assert start == date(2026, 5, 1)
        assert end == date(2026, 5, 1)

    def test_parse_invalid_no_colon(self):
        result = _parse_date_range("2026-04-25")
        assert result is None

    def test_parse_invalid_multiple_colons(self):
        result = _parse_date_range("2026-04-25:2026-05-02:2026-05-03")
        assert result is None

    def test_parse_invalid_date_format(self):
        result = _parse_date_range("04/25/2026:05/02/2026")
        assert result is None

    def test_parse_invalid_reversed_dates(self):
        result = _parse_date_range("2026-05-02:2026-04-25")
        assert result is None

    def test_parse_empty_string(self):
        result = _parse_date_range("")
        assert result is None

    def test_parse_none_string(self):
        result = _parse_date_range(None)
        assert result is None


class TestAgentLimitMapParser:
    def test_parse_valid_map(self):
        result = _parse_agent_limit_map("openai_news_feed=7,aws_ml_blog_feed=3")
        assert result == {"openai_news_feed": 7, "aws_ml_blog_feed": 3}

    def test_parse_empty_map(self):
        assert _parse_agent_limit_map(None) == {}
        assert _parse_agent_limit_map("") == {}

    def test_parse_rejects_invalid_chunk(self):
        try:
            _parse_agent_limit_map("openai_news_feed")
            assert False, "Expected ValueError"
        except ValueError:
            assert True

    def test_parse_rejects_non_integer(self):
        try:
            _parse_agent_limit_map("openai_news_feed=x")
            assert False, "Expected ValueError"
        except ValueError:
            assert True


class TestAgentLimitMapValidation:
    def test_validate_accepts_known_agents(self):
        _validate_agent_limit_map(
            {"openai_news_feed": 7, "aws_ml_blog_feed": 3},
            ["openai_news_feed", "aws_ml_blog_feed"],
        )

    def test_validate_rejects_unknown_agents(self):
        try:
            _validate_agent_limit_map(
                {"openai_news_feed": 7, "unknown_feed": 3},
                ["openai_news_feed", "aws_ml_blog_feed"],
            )
            assert False, "Expected ValueError"
        except ValueError as exc:
            message = str(exc)
            assert "unknown_feed" in message
            assert "openai_news_feed" in message


class TestAgentLimitAliasResolution:
    def test_resolve_short_aliases(self):
        result = _resolve_agent_limit_aliases(
            {"openai": 7, "aws": 3, "qwen_blog_feed": 2},
            {
                "openai": "openai_news_feed",
                "aws": "aws_ml_blog_feed",
            },
        )
        assert result == {
            "openai_news_feed": 7,
            "aws_ml_blog_feed": 3,
            "qwen_blog_feed": 2,
        }

    def test_resolve_rejects_duplicate_target(self):
        try:
            _resolve_agent_limit_aliases(
                {"openai": 7, "openai_news_feed": 5},
                {"openai": "openai_news_feed"},
            )
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "openai_news_feed" in str(exc)


class TestVendorScrapeFetchOnlyIntegration:
    def test_vendor_scrape_fetch_only_command_path(self, monkeypatch, capsys, tmp_path):
        from llmwatch.agents.watchers import vendor_scrape

        monkeypatch.setattr(
            vendor_scrape,
            "_HEALTH_CACHE_PATH",
            str(tmp_path / "vendor_scrape_health.json"),
        )

        class _FakeResp:
            def __init__(self, text: str, status_code: int = 200):
                self.text = text
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"http {self.status_code}")

        def _fake_get(url, **kwargs):
            today_text = "May 4, 2026"
            pages = {
                "https://ai.meta.com/blog/": f'<a href="/blog/meta-post">Meta Post</a> {today_text}',
                "https://www.anthropic.com/news": f'<a href="/news/anthropic-post">Anthropic Post</a> {today_text}',
                "https://mistral.ai/news": f'<a href="/news/mistral-post">Mistral Post</a> {today_text}',
                "https://x.ai/news": f'<a href="/news/xai-post">xAI Post</a> {today_text}',
            }
            return _FakeResp(f"<html><body>{pages.get(url, '')}</body></html>")

        monkeypatch.setattr(vendor_scrape.requests, "get", _fake_get)

        exit_code = cli_main(
            [
                "--vendor-scrape-fetch-only",
                "--vendor-scrape-lookback-days",
                "30",
                "--vendor-scrape-source-limits",
                "meta=1,anthropic=1,mistral=1,xai=1",
            ]
        )

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "meta_ai_blog_scrape: 1 item(s)" in out
        assert "anthropic_news_scrape: 1 item(s)" in out
        assert "mistral_news_scrape: 1 item(s)" in out
        assert "xai_news_scrape: 1 item(s)" in out
        assert "Total vendor scrape items: 4" in out

    def test_vendor_scrape_fetch_only_prints_health_warning_for_zero_items(
        self,
        monkeypatch,
        capsys,
        tmp_path,
    ):
        from llmwatch.agents.watchers import vendor_scrape

        monkeypatch.setenv("LLMWATCH_VENDOR_SCRAPE_HEALTH_WARNING_STREAK", "1")
        monkeypatch.setattr(
            vendor_scrape,
            "_HEALTH_CACHE_PATH",
            str(tmp_path / "vendor_scrape_health.json"),
        )

        class _FakeResp:
            def __init__(self, text: str, status_code: int = 200):
                self.text = text
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"http {self.status_code}")

        def _fake_get(url, **kwargs):
            today_text = "May 4, 2026"
            pages = {
                "https://ai.meta.com/blog/": f'<a href="/blog/meta-post">Meta Post</a> {today_text}',
                "https://www.anthropic.com/news": "<a href=\"/company\">Company</a>",
                "https://mistral.ai/news": f'<a href="/news/mistral-post">Mistral Post</a> {today_text}',
                "https://x.ai/news": f'<a href="/news/xai-post">xAI Post</a> {today_text}',
            }
            return _FakeResp(f"<html><body>{pages.get(url, '')}</body></html>")

        monkeypatch.setattr(vendor_scrape.requests, "get", _fake_get)

        exit_code = cli_main(
            [
                "--vendor-scrape-fetch-only",
                "--vendor-scrape-lookback-days",
                "30",
                "--vendor-scrape-source-limits",
                "meta=1,anthropic=1,mistral=1,xai=1",
            ]
        )

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "Scrape health warnings:" in out
        assert "anthropic_news_scrape: 0 items without request errors" in out

    def test_vendor_scrape_fetch_only_soft_fail_returns_success_on_errors(
        self,
        monkeypatch,
        capsys,
        tmp_path,
    ):
        from llmwatch.agents.watchers import vendor_scrape

        monkeypatch.setattr(
            vendor_scrape,
            "_HEALTH_CACHE_PATH",
            str(tmp_path / "vendor_scrape_health.json"),
        )

        class _FakeResp:
            def __init__(self, text: str, status_code: int = 200):
                self.text = text
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"http {self.status_code}")

        def _fake_get(url, **kwargs):
            if url == "https://ai.meta.com/blog/":
                raise vendor_scrape.requests.RequestException("challenge page detected")
            today_text = "May 4, 2026"
            pages = {
                "https://www.anthropic.com/news": f'<a href="/news/anthropic-post">Anthropic Post</a> {today_text}',
                "https://mistral.ai/news": f'<a href="/news/mistral-post">Mistral Post</a> {today_text}',
                "https://x.ai/news": f'<a href="/news/xai-post">xAI Post</a> {today_text}',
            }
            return _FakeResp(f"<html><body>{pages.get(url, '')}</body></html>")

        monkeypatch.setattr(vendor_scrape.requests, "get", _fake_get)

        exit_code = cli_main(
            [
                "--vendor-scrape-fetch-only",
                "--vendor-scrape-soft-fail",
                "--vendor-scrape-lookback-days",
                "30",
            ]
        )

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "meta_ai_blog_scrape: 0 item(s), 1 error(s)" in out
        assert "Soft-fail enabled: returning success despite scrape fetch errors." in out
