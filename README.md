# llm-watch

> Keep track and report on large language models.

A modular, agent-based system that monitors the world of large language models
and produces a **weekly investigation report** in Markdown.

## Docs Index

### Active Docs

- [README.md](README.md) - Project overview, usage, and architecture
- [docs/CONSOLIDATION_QUICK_REFERENCE.md](docs/CONSOLIDATION_QUICK_REFERENCE.md) - Canonical current-state consolidation behavior and roadmap
- [docs/llm_watch_report_2026-05-02.md](docs/llm_watch_report_2026-05-02.md) - Latest generated report snapshot (example output)

### Archived Docs (Historical)

- [docs/CONSOLIDATION_AGENT_INVESTIGATION.md](docs/CONSOLIDATION_AGENT_INVESTIGATION.md) - Design investigation and decision record before implementation
- [docs/PHASE_1_MVP_IMPLEMENTATION.md](docs/PHASE_1_MVP_IMPLEMENTATION.md) - Phase 1 implementation runbook used during initial rollout

## Architecture

```
llmwatch/
├── agents/
│   ├── base.py            # BaseAgent, AgentResult, AgentRegistry (registry singleton)
│   ├── watchers/
│   │   ├── huggingface.py        # Watches HuggingFace trending models
│   │   ├── huggingface_papers.py # Watches HuggingFace trending papers
│   │   ├── neuron_feed.py        # Watches The Neuron Atom feed (feed-first mode)
│   │   ├── ollama.py             # Watches the Ollama model library
│   │   ├── vendor_blogs.py       # Watches vendor AI/news blogs via RSS/Atom feeds
│   │   └── vendor_scrape.py      # Watches vendor pages via HTML scraping (Phase 2)
│   ├── lookup/
│   │   └── arxiv.py       # Looks up arXiv papers for discovered models
│   └── reporter.py        # Aggregates findings into a weekly Markdown report
├── orchestrator.py        # Runs agents in three phases: watch → lookup → report
└── main.py                # CLI entry point
```

### Agent phases

| Phase      | Agents           | Description                                                  |
|------------|------------------|--------------------------------------------------------------|
| `watcher`  | HuggingFace Models, HuggingFace Papers, Ollama, The Neuron, Vendor Blog Feeds, Vendor Scrape Sources | Fetch trending/new models/papers from public sources |
| `lookup`   | arXiv            | Search for papers related to discovered models               |
| `reporter` | StoryConsolidator, WeeklyReporter | Sequential reporter pipeline: consolidate stories, then render report |

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Generate a weekly report (written to the current directory)
llm-watch

# Print the report to stdout without writing a file
llm-watch --dry-run

# List all registered agents
llm-watch --list-agents

# Run sequentially with verbose logging
llm-watch --no-parallel --verbose

# Write report to a custom directory
llm-watch --output-dir /path/to/reports

# Force fresh arXiv lookups (ignore cached lookup results)
llm-watch --arxiv-force-fetch

# Fetch and cache TLDR updates only (no report generation)
llm-watch --tldr-fetch-only

# Fetch only vendor blog feeds (no report generation)
llm-watch --vendor-blogs-fetch-only

# Fetch only vendor scrape sources (no report generation)
llm-watch --vendor-scrape-fetch-only

# Fetch scrape sources but do not fail process on per-source scrape errors
llm-watch --vendor-scrape-fetch-only --vendor-scrape-soft-fail

# Fetch TLDR data for a specific date range and merge with cache
llm-watch --tldr-fetch-only --tldr-date-range "2026-04-25:2026-05-02"

# Set Last Week in AI podcast lookback window (days)
llm-watch --lwiai-lookback-days 14

# Set The Neuron feed lookback window (days)
llm-watch --neuron-lookback-days 7

# Set vendor blog feed lookback window (days)
llm-watch --vendor-blog-lookback-days 14

# Set a global maximum items per vendor feed
llm-watch --vendor-blog-max-items 15

# Set per-feed limits (agent_name=max_items)
llm-watch --vendor-blog-feed-limits "openai_news_feed=10,aws_ml_blog_feed=5"

# Use short aliases for common feeds
llm-watch --vendor-blog-feed-limits "openai=10,aws=5,google=4"

# Set vendor scrape lookback and limits
llm-watch --vendor-scrape-lookback-days 14 --vendor-scrape-max-items 10

# Per-source scrape limits with aliases
llm-watch --vendor-scrape-source-limits "meta=8,anthropic=6,mistral=6,xai=6"
```

Or run as a module:

```bash
python -m llmwatch.main [options]
```

### arXiv Lookup Caching

The arXiv lookup agent uses a local cache by default to reduce timeouts and
network calls.

- Cache file: `.llmwatch_cache/arxiv_lookup_cache.json`
- Default behavior: look up each normalized query term in cache first
- Override behavior: use `--arxiv-force-fetch` to bypass cache and fetch all
   terms directly from arXiv

### TLDR Local Hybrid Filter (Ollama)

TLDR items use a hybrid classifier:

- Primary: local Ollama classification for whether an item belongs in
   `Trending & New Models`
- Fallback: keyword-based rules when Ollama is unavailable or returns invalid
   output

TLDR items are cached so daily fetches can be merged into less-frequent reports.

- Cache file: `.llmwatch_cache/tldr_items.json`
- Fetch-only mode: `--tldr-fetch-only` updates the TLDR cache without generating
  a report
- Weekly/full report runs read merged cached TLDR items from recent history

#### Fetching TLDR Data for a Date Range

Use `--tldr-date-range` with `--tldr-fetch-only` to backfill TLDR data from
previous days:

```bash
# Fetch and cache TLDR data from April 25 to May 2
llm-watch --tldr-fetch-only --tldr-date-range "2026-04-25:2026-05-02"

# Output: "TLDR cache updated: 50 item(s) for date range 2026-04-25 to 2026-05-02."
```

The date range format is ISO 8601 (`YYYY-MM-DD:YYYY-MM-DD`):
- Start and end dates are both **inclusive**
- The range skips dates where no edition is published (404 or redirect responses)
- All fetched items are merged with the existing cache and deduplicated by URL
- Cache retention is still governed by `LLMWATCH_TLDR_HISTORY_DAYS`

Use this feature to catch up on missed daily updates or to backfill the cache
when starting fresh.

Environment knobs:

- `LLMWATCH_TLDR_OLLAMA_FILTER` (default: enabled)
- `LLMWATCH_TLDR_FILTER_MODEL` (default: `llama3.2:3b`)
- `LLMWATCH_OLLAMA_API_URL` (default: `http://localhost:11434/api/generate`)
- `LLMWATCH_TLDR_HISTORY_DAYS` (default: `14`)

### Last Week in AI Podcast Link Filtering

The `lastweekinai_podcast` watcher extracts podcast summary links from the
public feed and applies quality filters to keep high-signal links:

- Keeps: likely article/research links (for example, news posts and arXiv URLs)
- Drops: low-signal links such as social/profile URLs and generic non-article pages

Use `--lwiai-lookback-days` to control how far back podcast episodes are scanned
on a normal report run.

### The Neuron Feed-First Watcher

The `neuron_feed` watcher reads The Neuron Atom feed directly instead of scraping
full post pages. This mode is robust when post pages are protected by anti-bot
challenges.

- Includes: post title, URL, publication date, category term, and summary
- Categories tracked by default: `newsletter`, `explainer-articles`
- Lookback control: `--neuron-lookback-days`

### Vendor Blog Feed Watchers (Phase 1)

The `vendor_blogs` watcher module adds feed-first watchers for these sources:

- OpenAI News
- Google AI Blog
- Google DeepMind Blog
- Microsoft AI Blog
- AWS Machine Learning Blog
- Qwen Blog (legacy Hugo feed)

These are intentionally RSS/Atom-first to keep maintenance low and avoid brittle
HTML scraping.

- Lookback control on normal runs: `--vendor-blog-lookback-days`
- Global per-feed cap: `--vendor-blog-max-items`
- Optional per-feed overrides: `--vendor-blog-feed-limits "agent_name=n,..."`
- Supported short aliases: `openai`, `google`, `deepmind`, `microsoft`, `aws`, `qwen`
- Feed-only run mode: `--vendor-blogs-fetch-only`

### Vendor Scrape Watchers (Phase 2)

The `vendor_scrape` watcher module adds lightweight listing-page scraping for:

- Meta AI Blog
- Anthropic News
- Mistral News
- xAI News

These are used when stable RSS/Atom feeds are unavailable.

- Lookback control on normal runs: `--vendor-scrape-lookback-days`
- Global per-source cap: `--vendor-scrape-max-items`
- Optional per-source overrides: `--vendor-scrape-source-limits "agent_name=n,..."`
- Supported short aliases: `meta`, `anthropic`, `mistral`, `xai`
- Scrape-only run mode: `--vendor-scrape-fetch-only`
- Optional scrape-only soft fail: `--vendor-scrape-soft-fail`

Environment knobs:

- `LLMWATCH_VENDOR_SCRAPE_HEALTH_WARNING_STREAK` (default: `2`)
   - Emit scrape health warnings only after N consecutive zero-item runs.

### Source Operations

Use these commands as a quick operational checklist:

```bash
# Feed-only health check
llm-watch --vendor-blogs-fetch-only --vendor-blog-lookback-days 14

# Scrape-only health check
llm-watch --vendor-scrape-fetch-only --vendor-scrape-lookback-days 14

# Full report run (end-to-end)
llm-watch --no-parallel
```

### Scrape Troubleshooting

If scrape sources degrade, use this table to triage quickly:

| Symptom | Likely Cause | Suggested Action |
|---|---|---|
| `0 items` with no request errors (single run) | No recent posts or transient layout mismatch | Re-run with larger lookback window |
| `0 items` warning after streak threshold | Persistent layout drift or anti-bot challenge | Inspect listing HTML and update source regex rules |
| Request failures/timeouts | Network instability or temporary endpoint issues | Retry run; lower concurrency via `--no-parallel` |
| Scraped links look low-signal/navigation-only | Generic links matching source pattern | Tighten per-source filtering rules in `vendor_scrape.py` |

### Common Links Ranking and Suppression

The `story_consolidator` and `weekly_reporter` pipeline builds a `Common Links`
section that prioritizes useful repeated links while suppressing noisy repeats.

- Ranking signal combines:
   - source-class diversity (for example newsletter + podcast)
   - source diversity and appearance count
   - freshness and novelty adjustments
- Buckets rendered in report:
   - `High Signal Common Links`
   - `Repeated References`
   - `Suppressed Repeated Links`
- Link metadata shown in report includes `Type` and `Coverage`

Environment knobs:

- `LLMWATCH_CONSOLIDATOR_SIMILARITY_THRESHOLD` (default: `0.85`)
- `LLMWATCH_CONSOLIDATOR_TEMPORAL_WINDOW_DAYS` (default: `7`)
- `LLMWATCH_CONSOLIDATOR_SUPPRESS_SPONSORS` (default: `true`)
- `LLMWATCH_CONSOLIDATOR_SUPPRESS_SOCIAL_SINGLE_SOURCE` (default: `true`)
- `LLMWATCH_CONSOLIDATOR_SUPPRESS_DOMAINS` (default: empty, comma-separated)
- `LLMWATCH_CONSOLIDATOR_ALLOW_DOMAINS` (default: empty, comma-separated; overrides suppression)

## Adding a new agent

1. Create a new file in `llmwatch/agents/watchers/` (or `lookup/`).
2. Subclass `BaseAgent` and set `name` and `category`.
3. Implement the `run(self, context)` method.
4. Call `registry.register(MyAgent())` at module level.
5. Import the module in `llmwatch/main.py`.

Example:

```python
# llmwatch/agents/watchers/my_source.py
from llmwatch.agents.base import BaseAgent, registry

class MySourceWatcher(BaseAgent):
    name = "my_source"
    category = "watcher"

    def run(self, context=None):
        # ... fetch data from your source ...
        return self._result(data=[{"model_id": "org/model", "url": "..."}])

registry.register(MySourceWatcher())
```

## Testing

```bash
pytest
```

## Report format

Each run produces a Markdown file named `llm_watch_report_YYYY-MM-DD.md` with
five sections:

1. **Common Links This Week** – consolidated cross-source links ranked by
   cross-source signal, split into high-signal, repeated-reference, and
   suppressed-link subsections.
2. **Trending & New Models** – models discovered by watcher agents, grouped by
   source.
3. **Related Research Papers (arXiv)** – papers found by the lookup agent,
   grouped by search term.
4. **Warnings** – any errors that occurred during the run.
5. **New Sources Discovered** – external URLs found in model descriptions or
   paper abstracts that are not yet tracked, highlighted for future monitoring.
