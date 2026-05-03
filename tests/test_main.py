"""
Unit tests for CLI argument parsing in llmwatch.main.
"""

from datetime import date
from llmwatch.main import _build_parser, _parse_date_range


class TestCliParser:
    def test_parses_tldr_fetch_only_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--tldr-fetch-only"])
        assert args.tldr_fetch_only is True

    def test_tldr_fetch_only_defaults_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.tldr_fetch_only is False

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
