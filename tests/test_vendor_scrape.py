"""Unit tests for Phase 2 vendor scrape watchers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from llmwatch.agents.watchers import vendor_scrape as scrape_mod


class _FakeResp:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"http {self.status_code}")


def _meta_watcher() -> scrape_mod.VendorScrapeWatcher:
    cfg = next(c for c in scrape_mod._VENDOR_SCRAPE_CONFIGS if c.agent_name == "meta_ai_blog_scrape")
    return scrape_mod.VendorScrapeWatcher(cfg)


def _anthropic_watcher() -> scrape_mod.VendorScrapeWatcher:
    cfg = next(
        c for c in scrape_mod._VENDOR_SCRAPE_CONFIGS if c.agent_name == "anthropic_news_scrape"
    )
    return scrape_mod.VendorScrapeWatcher(cfg)


def _mistral_watcher() -> scrape_mod.VendorScrapeWatcher:
    cfg = next(c for c in scrape_mod._VENDOR_SCRAPE_CONFIGS if c.agent_name == "mistral_news_scrape")
    return scrape_mod.VendorScrapeWatcher(cfg)


def _xai_watcher() -> scrape_mod.VendorScrapeWatcher:
    cfg = next(c for c in scrape_mod._VENDOR_SCRAPE_CONFIGS if c.agent_name == "xai_news_scrape")
    return scrape_mod.VendorScrapeWatcher(cfg)


def test_meta_scrape_watcher_filters_by_lookback(monkeypatch):
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=1)).strftime("%B %d, %Y")
    old = (now - timedelta(days=40)).strftime("%B %d, %Y")

    html = f"""
    <html><body>
      <div>{recent} <a href="/blog/recent-meta-post/">Recent Meta Post</a></div>
      <div>{old} <a href="/blog/old-meta-post/">Old Meta Post</a></div>
      <div><a href="/about/">About</a></div>
    </body></html>
    """

    monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(html))

    result = _meta_watcher().run(context={"vendor_scrape_lookback_days": 14})

    assert result.ok()
    assert len(result.data) == 1
    assert result.data[0]["model_id"] == "Recent Meta Post"
    assert result.data[0]["source"] == "meta_ai_blog"


def test_anthropic_scrape_per_source_limit_overrides_global(monkeypatch):
    now = datetime.now(timezone.utc)
    d1 = (now - timedelta(days=1)).strftime("%b %d, %Y")
    d2 = (now - timedelta(days=2)).strftime("%b %d, %Y")

    html = f"""
    <html><body>
      <div>{d1} <a href="/news/first-anthropic-post">First Anthropic Post</a></div>
      <div>{d2} <a href="/news/second-anthropic-post">Second Anthropic Post</a></div>
    </body></html>
    """

    monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(html))

    result = _anthropic_watcher().run(
        context={
            "vendor_scrape_lookback_days": 30,
            "vendor_scrape_max_items": 10,
            "vendor_scrape_per_source_max_items": {"anthropic_news_scrape": 1},
        }
    )

    assert result.ok()
    assert len(result.data) == 1
    assert result.data[0]["model_id"] == "First Anthropic Post"


def test_vendor_scrape_watcher_handles_request_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.RequestException("scrape unavailable")

    monkeypatch.setattr(scrape_mod.requests, "get", _raise)

    result = _meta_watcher().run()
    assert not result.ok()
    assert result.data == []
    assert any("scrape unavailable" in err for err in result.errors)


def test_mistral_scrape_watcher_extracts_news_links(monkeypatch):
        html = """
        <html><body>
            <div>May 1, 2026 <a href="/news/mistral-medium-update">Mistral Medium Update</a></div>
            <div><a href="/about">About</a></div>
        </body></html>
        """

        monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(html))
        result = _mistral_watcher().run(context={"vendor_scrape_lookback_days": 30})

        assert result.ok()
        assert len(result.data) == 1
        assert result.data[0]["url"] == "https://mistral.ai/news/mistral-medium-update"


def test_xai_scrape_watcher_extracts_news_links(monkeypatch):
    html = """
    <html><body>
      <div>Apr 30, 2026 <a href="/news/grok-update">Grok Update</a></div>
      <div><a href="/grok">Grok</a></div>
    </body></html>
    """

    monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(html))
    result = _xai_watcher().run(context={"vendor_scrape_lookback_days": 30})

    assert result.ok()
    assert len(result.data) == 1
    assert result.data[0]["url"] == "https://x.ai/news/grok-update"


def test_vendor_scrape_health_warning_on_zero_items(monkeypatch, caplog, tmp_path):
    caplog.set_level("WARNING")
    monkeypatch.setattr(
        scrape_mod,
        "_HEALTH_CACHE_PATH",
        str(tmp_path / "vendor_scrape_health.json"),
    )
    monkeypatch.setattr(scrape_mod, "_CACHE_DIR", str(tmp_path))
    html = "<html><body><a href=\"/about\">About</a></body></html>"
    monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(html))

    result = _meta_watcher().run(
        context={
            "vendor_scrape_lookback_days": 30,
            "vendor_scrape_health_warning_streak": 1,
        }
    )

    assert result.ok()
    assert result.data == []
    assert "scrape health warning" in caplog.text


def test_vendor_scrape_retries_challenge_then_succeeds(monkeypatch):
    calls = {"count": 0}

    def _fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResp("<html><title>Just a moment...</title></html>")
        return _FakeResp(
            '<html><body><div>May 1, 2026 <a href="/news/mistral-medium-update">Mistral Medium Update</a></div></body></html>'
        )

    monkeypatch.setattr(scrape_mod.requests, "get", _fake_get)
    result = _mistral_watcher().run(context={"vendor_scrape_lookback_days": 30, "vendor_scrape_retry_attempts": 2})

    assert result.ok()
    assert calls["count"] == 2
    assert len(result.data) == 1


def test_vendor_scrape_health_streak_persists_and_resets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        scrape_mod,
        "_HEALTH_CACHE_PATH",
        str(tmp_path / "vendor_scrape_health.json"),
    )
    monkeypatch.setattr(scrape_mod, "_CACHE_DIR", str(tmp_path))

    # First run: zero items increments streak to 1.
    monkeypatch.setattr(
        scrape_mod.requests,
        "get",
        lambda *args, **kwargs: _FakeResp("<html><body><a href=\"/about\">About</a></body></html>"),
    )
    _meta_watcher().run(context={"vendor_scrape_lookback_days": 30})
    assert scrape_mod.get_health_streak("meta_ai_blog_scrape") == 1

    # Second run: successful scrape resets streak.
    monkeypatch.setattr(
        scrape_mod.requests,
        "get",
        lambda *args, **kwargs: _FakeResp(
            '<html><body><div>May 4, 2026 <a href="/blog/meta-post">Meta Post</a></div></body></html>'
        ),
    )
    _meta_watcher().run(context={"vendor_scrape_lookback_days": 30})
    assert scrape_mod.get_health_streak("meta_ai_blog_scrape") == 0


def test_anthropic_title_cleanup_from_noisy_anchor_text(monkeypatch):
        html = """
        <html><body>
            <a href="/news/claude-opus-4-7">
                Introducing Claude Opus 4.7ProductApr 16, 2026Our latest Opus model brings stronger performance.
            </a>
        </body></html>
        """

        monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(html))
        result = _anthropic_watcher().run(context={"vendor_scrape_lookback_days": 365})

        assert result.ok()
        assert len(result.data) == 1
        assert result.data[0]["model_id"] == "Claude Opus 4 7"


def test_meta_title_prefers_descriptive_anchor_over_featured(monkeypatch):
        html = """
        <html><body>
            <div>May 1, 2026 <a href="/blog/introducing-muse-spark-msl/">FEATURED</a></div>
            <div>May 1, 2026 <a href="/blog/introducing-muse-spark-msl/">Introducing Muse Spark: Scaling Towards Personal Superintelligence</a></div>
        </body></html>
        """

        monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(html))
        result = _meta_watcher().run(context={"vendor_scrape_lookback_days": 365})

        assert result.ok()
        assert len(result.data) == 1
        assert result.data[0]["model_id"] == "Introducing Muse Spark: Scaling Towards Personal Superintelligence"


def test_duplicate_url_prefers_higher_quality_title(monkeypatch):
        html = """
        <html><body>
            <div>May 1, 2026 <a href="/news/claude-opus-4-7">Claude Update</a></div>
            <div>May 1, 2026 <a href="/news/claude-opus-4-7">Introducing Claude Opus 4.7: Better Coding and Agent Reliability</a></div>
        </body></html>
        """

        monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(html))
        result = _anthropic_watcher().run(context={"vendor_scrape_lookback_days": 365})

        assert result.ok()
        assert len(result.data) == 1
        assert result.data[0]["model_id"] == "Introducing Claude Opus 4.7: Better Coding and Agent Reliability"


def test_duplicate_url_prefers_longer_prefix_title(monkeypatch):
        html = """
        <html><body>
            <div>May 1, 2026 <a href="/news/claude-opus-4-7">Claude Opus</a></div>
            <div>May 1, 2026 <a href="/news/claude-opus-4-7">Claude Opus 4.7 Extended</a></div>
        </body></html>
        """

        monkeypatch.setattr(scrape_mod.requests, "get", lambda *args, **kwargs: _FakeResp(html))
        result = _anthropic_watcher().run(context={"vendor_scrape_lookback_days": 365})

        assert result.ok()
        assert len(result.data) == 1
        assert result.data[0]["model_id"] == "Claude Opus 4.7 Extended"
