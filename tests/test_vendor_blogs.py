"""Unit tests for vendor blog feed watchers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from llmwatch.agents.watchers import vendor_blogs as vendor_mod


class _FakeResp:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"http {self.status_code}")


def _openai_watcher() -> vendor_mod.VendorBlogFeedWatcher:
    cfg = next(c for c in vendor_mod._VENDOR_FEEDS if c.agent_name == "openai_news_feed")
    return vendor_mod.VendorBlogFeedWatcher(cfg)


def _qwen_watcher() -> vendor_mod.VendorBlogFeedWatcher:
    cfg = next(c for c in vendor_mod._VENDOR_FEEDS if c.agent_name == "qwen_blog_feed")
    return vendor_mod.VendorBlogFeedWatcher(cfg)


def test_vendor_blog_rss_watcher_filters_by_lookback(monkeypatch):
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    old = (now - timedelta(days=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")

    rss = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <rss version=\"2.0\">
      <channel>
        <title>OpenAI News</title>
        <item>
          <title>Recent OpenAI update</title>
          <link>https://openai.com/index/recent-update/</link>
          <description>Recent details</description>
          <pubDate>{recent}</pubDate>
          <category>Product</category>
        </item>
        <item>
          <title>Old OpenAI update</title>
          <link>https://openai.com/index/old-update/</link>
          <description>Old details</description>
          <pubDate>{old}</pubDate>
          <category>Company</category>
        </item>
      </channel>
    </rss>
    """

    monkeypatch.setattr(
        vendor_mod.requests,
        "get",
        lambda *args, **kwargs: _FakeResp(rss),
    )

    result = _openai_watcher().run(context={"vendor_blog_lookback_days": 7})

    assert result.ok()
    assert len(result.data) == 1
    item = result.data[0]
    assert item["model_id"] == "Recent OpenAI update"
    assert item["source"] == "openai_news"
    assert "product" in item["tags"]


def test_vendor_blog_atom_watcher_parses_entries(monkeypatch):
    now = datetime.now(timezone.utc)
    published = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")

    atom = f"""<?xml version=\"1.0\" encoding=\"utf-8\"?>
    <feed xmlns=\"http://www.w3.org/2005/Atom\">
      <title>Qwen Blog</title>
      <entry>
        <title>Qwen update</title>
        <link href=\"https://qwenlm.github.io/blog/qwen-update/\" rel=\"alternate\"/>
        <summary>Qwen summary</summary>
        <published>{published}</published>
        <category term=\"Open-Source\"/>
      </entry>
    </feed>
    """

    monkeypatch.setattr(
        vendor_mod.requests,
        "get",
        lambda *args, **kwargs: _FakeResp(atom),
    )

    result = _qwen_watcher().run(context={"vendor_blog_lookback_days": 14})

    assert result.ok()
    assert len(result.data) == 1
    item = result.data[0]
    assert item["model_id"] == "Qwen update"
    assert item["url"] == "https://qwenlm.github.io/blog/qwen-update/"
    assert item["source"] == "qwen_blog"
    assert "open-source" in item["tags"]


def test_vendor_blog_watcher_handles_request_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.RequestException("feed unavailable")

    monkeypatch.setattr(vendor_mod.requests, "get", _raise)

    result = _openai_watcher().run()
    assert not result.ok()
    assert result.data == []
    assert any("feed unavailable" in err for err in result.errors)


def test_vendor_blog_per_feed_limit_overrides_global_limit(monkeypatch):
    now = datetime.now(timezone.utc)
    recent_a = (now - timedelta(days=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    recent_b = (now - timedelta(days=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")

    rss = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <rss version=\"2.0\">
      <channel>
        <title>OpenAI News</title>
        <item>
          <title>Recent A</title>
          <link>https://openai.com/index/recent-a/</link>
          <description>Recent details A</description>
          <pubDate>{recent_a}</pubDate>
        </item>
        <item>
          <title>Recent B</title>
          <link>https://openai.com/index/recent-b/</link>
          <description>Recent details B</description>
          <pubDate>{recent_b}</pubDate>
        </item>
      </channel>
    </rss>
    """

    monkeypatch.setattr(
        vendor_mod.requests,
        "get",
        lambda *args, **kwargs: _FakeResp(rss),
    )

    result = _openai_watcher().run(
        context={
            "vendor_blog_lookback_days": 30,
            "vendor_blog_max_items": 5,
            "vendor_blog_per_feed_max_items": {"openai_news_feed": 1},
        }
    )

    assert result.ok()
    assert len(result.data) == 1
    assert result.data[0]["model_id"] == "Recent A"
