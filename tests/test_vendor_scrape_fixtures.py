"""Fixture-based scrape parser tests to detect layout drift regressions."""

from __future__ import annotations

from pathlib import Path

from llmwatch.agents.watchers import vendor_scrape as scrape_mod


class _FakeResp:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def _watcher(agent_name: str) -> scrape_mod.VendorScrapeWatcher:
    cfg = next(c for c in scrape_mod._VENDOR_SCRAPE_CONFIGS if c.agent_name == agent_name)
    return scrape_mod.VendorScrapeWatcher(cfg)


def _fixture_text(name: str) -> str:
    root = Path(__file__).parent / "fixtures" / "vendor_scrape"
    return (root / name).read_text(encoding="utf-8")


def test_meta_fixture_extracts_expected_item(monkeypatch):
    monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(_fixture_text("meta_listing.html")))
    result = _watcher("meta_ai_blog_scrape").run(context={"vendor_scrape_lookback_days": 365})
    assert result.ok()
    assert len(result.data) == 1
    assert result.data[0]["url"] == "https://ai.meta.com/blog/introducing-muse-spark-msl/"


def test_meta_mixed_anchor_fixture_prefers_descriptive_title(monkeypatch):
    monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(_fixture_text("meta_mixed_anchors.html")))
    result = _watcher("meta_ai_blog_scrape").run(context={"vendor_scrape_lookback_days": 365})
    assert result.ok()
    assert len(result.data) == 1
    assert result.data[0]["model_id"] == "Introducing Muse Spark: Scaling Towards Personal Superintelligence"


def test_anthropic_fixture_extracts_expected_items(monkeypatch):
    monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(_fixture_text("anthropic_listing.html")))
    result = _watcher("anthropic_news_scrape").run(context={"vendor_scrape_lookback_days": 365})
    assert result.ok()
    assert len(result.data) == 2


def test_mistral_fixture_extracts_expected_item(monkeypatch):
    monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(_fixture_text("mistral_listing.html")))
    result = _watcher("mistral_news_scrape").run(context={"vendor_scrape_lookback_days": 365})
    assert result.ok()
    assert len(result.data) == 1
    assert result.data[0]["url"] == "https://mistral.ai/news/mistral-small-4"


def test_xai_fixture_extracts_expected_item(monkeypatch):
    monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(_fixture_text("xai_listing.html")))
    result = _watcher("xai_news_scrape").run(context={"vendor_scrape_lookback_days": 365})
    assert result.ok()
    assert len(result.data) == 1
    assert result.data[0]["url"] == "https://x.ai/news/grok-custom-voices"
