"""Unit tests for The Neuron feed watcher."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from llmwatch.agents.watchers import neuron_feed as neuron_mod


class _FakeResp:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"http {self.status_code}")


def _entry(title: str, link: str, published: str, category: str, summary: str = "") -> str:
    summary_block = f"<summary>{summary}</summary>" if summary else ""
    return f"""
    <entry>
      <title>{title}</title>
      <link href="{link}"/>
      {summary_block}
      <category term="{category}"/>
      <published>{published}</published>
    </entry>
    """


def _feed(entries: list[str]) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>TheNeuron</title>
      {''.join(entries)}
    </feed>
    """


def test_neuron_feed_watcher_collects_recent_items(monkeypatch):
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    old = (now - timedelta(days=20)).isoformat().replace("+00:00", "Z")

    feed_xml = _feed(
        [
            _entry(
                title="Around the Horn Digest",
                link="https://www.theneuron.ai/explainer-articles/a/",
                published=recent,
                category="explainer-articles",
                summary="Daily roundup",
            ),
            _entry(
                title="Newsletter Issue",
                link="https://www.theneuron.ai/newsletter/b/",
                published=recent,
                category="newsletter",
            ),
            _entry(
                title="Old Item",
                link="https://www.theneuron.ai/newsletter/old/",
                published=old,
                category="newsletter",
            ),
            _entry(
                title="Unrelated",
                link="https://www.theneuron.ai/other/c/",
                published=recent,
                category="other",
            ),
        ]
    )

    monkeypatch.setattr(
        neuron_mod.requests,
        "get",
        lambda *args, **kwargs: _FakeResp(feed_xml),
    )

    watcher = neuron_mod.NeuronFeedWatcher()
    result = watcher.run(context={"neuron_lookback_days": 7})

    assert result.ok()
    assert len(result.data) == 2
    assert any(item["neuron_category"] == "explainer-articles" for item in result.data)
    assert any(item["neuron_category"] == "newsletter" for item in result.data)


def test_neuron_feed_watcher_handles_request_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.RequestException("feed unavailable")

    monkeypatch.setattr(neuron_mod.requests, "get", _raise)

    watcher = neuron_mod.NeuronFeedWatcher()
    result = watcher.run()

    assert not result.ok()
    assert result.data == []
    assert any("feed unavailable" in err for err in result.errors)
