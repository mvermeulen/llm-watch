"""Unit tests for the Last Week in AI podcast watcher."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llmwatch.agents.watchers import lastweekinai_podcast as lwiai_mod


def _rss_with_items(items: list[str]) -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:content="http://purl.org/rss/1.0/modules/content/" version="2.0">
  <channel>
    <title>Last Week in AI</title>
    {items}
  </channel>
</rss>
""".format(items="\n".join(items))


def _podcast_item(title: str, pub_date: str, link: str, description: str, content_html: str) -> str:
    return f"""
<item>
  <title><![CDATA[{title}]]></title>
  <description><![CDATA[{description}]]></description>
  <link>{link}</link>
  <pubDate>{pub_date}</pubDate>
  <content:encoded><![CDATA[{content_html}]]></content:encoded>
</item>
"""


class _FakeResp:
    def __init__(self, text: str = "", url: str = "", status_code: int = 200):
        self.text = text
        self.url = url
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def test_watcher_collects_recent_episode_and_links(monkeypatch):
    now = datetime.now(timezone.utc)
    pub_date = (now - timedelta(days=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")

    feed_xml = _rss_with_items(
        [
            _podcast_item(
                title="LWiAI Podcast #999 - Test Episode",
                pub_date=pub_date,
                link="https://lastweekin.ai/p/lwiai-podcast-999-test",
                description="Key updates across AI this week.",
                content_html=(
                    '<p>(00:10:00) <a href="https://example.com/story">Example Story</a></p>'
                    '<p><a href="https://lastweekin.ai/p/internal">Read more</a></p>'
                ),
            )
        ]
    )

    def fake_get(url, *args, **kwargs):
        if url == lwiai_mod._FEED_URL:
            return _FakeResp(text=feed_xml, url=url)
        if url == "https://example.com/story":
            return _FakeResp(
                text="<html><head><title>Example Story Title</title></head></html>",
                url=url,
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(lwiai_mod.requests, "get", fake_get)

    watcher = lwiai_mod.LastWeekInAIPodcastWatcher()
    result = watcher.run()

    assert result.ok()
    assert any(item.get("tags") == ["podcast_summary"] for item in result.data)

    link_items = [d for d in result.data if "podcast_link" in d.get("tags", [])]
    assert len(link_items) == 1
    assert link_items[0]["model_id"] == "Example Story"
    assert link_items[0]["url"] == "https://example.com/story"


def test_watcher_skips_episodes_outside_lookback(monkeypatch):
    now = datetime.now(timezone.utc)
    old_pub_date = (now - timedelta(days=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")

    feed_xml = _rss_with_items(
        [
            _podcast_item(
                title="LWiAI Podcast #900 - Old Episode",
                pub_date=old_pub_date,
                link="https://lastweekin.ai/p/old",
                description="Old summary",
                content_html='<a href="https://example.com/old">Old</a>',
            )
        ]
    )

    monkeypatch.setattr(
        lwiai_mod.requests,
        "get",
        lambda url, *args, **kwargs: _FakeResp(text=feed_xml, url=url),
    )

    watcher = lwiai_mod.LastWeekInAIPodcastWatcher()
    result = watcher.run(context={"lwiai_lookback_days": 7})

    assert result.ok()
    assert result.data == []


def test_quality_filter_keeps_article_links_and_skips_social_profiles(monkeypatch):
    now = datetime.now(timezone.utc)
    pub_date = (now - timedelta(days=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")

    feed_xml = _rss_with_items(
        [
            _podcast_item(
                title="LWiAI Podcast #901 - Link Filter Episode",
                pub_date=pub_date,
                link="https://lastweekin.ai/p/lwiai-podcast-901-links",
                description="Testing links",
                content_html=(
                    '<a href="https://x.com/andrey_kurenkov">Andrey</a>'
                    '<a href="https://www.linkedin.com/in/jeremieharris/">Jeremie</a>'
                    '<a href="https://techcrunch.com/2026/04/21/sample-article/">Article</a>'
                ),
            )
        ]
    )

    def fake_get(url, *args, **kwargs):
        if url == lwiai_mod._FEED_URL:
            return _FakeResp(text=feed_xml, url=url)
        if "techcrunch.com" in url:
            return _FakeResp(
                text="<html><head><title>Sample TechCrunch Article</title></head></html>",
                url=url,
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(lwiai_mod.requests, "get", fake_get)

    watcher = lwiai_mod.LastWeekInAIPodcastWatcher()
    result = watcher.run()

    link_items = [d for d in result.data if "podcast_link" in d.get("tags", [])]
    assert len(link_items) == 1
    assert "techcrunch.com" in link_items[0]["url"]
    assert all("x.com" not in item["url"] for item in link_items)
    assert all("linkedin.com" not in item["url"] for item in link_items)