# llm-watch

> Keep track and report on large language models.

A modular, agent-based system that monitors the world of large language models
and produces a **weekly investigation report** in Markdown.

## Docs Index

### Active Docs

- [README.md](README.md) - Project overview, usage, and architecture
- [CHANGELOG.md](CHANGELOG.md) - Release notes and notable project changes
- [docs/CONSOLIDATION_QUICK_REFERENCE.md](docs/CONSOLIDATION_QUICK_REFERENCE.md) - Canonical current-state consolidation behavior and roadmap
- [docs/EDITOR_QUICK_REFERENCE.md](docs/EDITOR_QUICK_REFERENCE.md) - Canonical current-state editor behavior and deferred follow-up work
- [llm_watch_report_2026-05-05.md](llm_watch_report_2026-05-05.md) - Latest generated report snapshot (example output)

### Archived Docs (Historical)

- [docs/CONSOLIDATION_AGENT_INVESTIGATION.md](docs/CONSOLIDATION_AGENT_INVESTIGATION.md) - Design investigation and decision record before implementation
- [docs/PHASE_1_MVP_IMPLEMENTATION.md](docs/PHASE_1_MVP_IMPLEMENTATION.md) - Phase 1 implementation runbook used during initial rollout

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes and notable changes.

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
│   ├── reporter.py        # Aggregates findings into a weekly Markdown report
│   └── editor.py          # Post-processes the report via a local Ollama LLM (Phase 4)
├── ollama_client.py       # Shared Ollama HTTP client (used by editor and TLDR filter)
├── orchestrator.py        # Runs agents in four phases: watch → lookup → report → edit
└── main.py                # CLI entry point
```

### Agent phases

| Phase      | Agents           | Description                                                  |
|------------|------------------|--------------------------------------------------------------|
| `watcher`  | HuggingFace Models, HuggingFace Papers, Ollama, The Neuron, Vendor Blog Feeds, Vendor Scrape Sources | Fetch trending/new models/papers from public sources |
| `lookup`   | arXiv            | Search for papers related to discovered models               |
| `reporter` | StoryConsolidator, WeeklyReporter | Sequential reporter pipeline: consolidate stories, then render report |
| `editor`   | OllamaEditorAgent | Optional Phase 4: refine the finished Markdown report with a local Ollama model |

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

# Run the Ollama editor pass after generating the report
llm-watch --edit

# Run with a specific editor model
llm-watch --edit --editor-model qwen3:8b

# Skip certain editor tasks (comma-separated)
llm-watch --edit --editor-skip-tasks themes,model_digest

# Enable the editor via environment variable (no CLI flag needed)
LLMWATCH_EDITOR_ENABLED=true llm-watch

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

# Mark specific URLs as read so they are suppressed in future reports
llm-watch --mark-read https://example.com/story-a https://example.com/story-b

# Mark every link from a generated report as read
llm-watch --mark-read-from-report llm_watch_report_2026-05-05.md

# Mark only links inside a specific H2 section
llm-watch --mark-read-from-report llm_watch_report_2026-05-05.md --section "Common Links"

# Inspect or reset read-tracking state
llm-watch --list-read
llm-watch --unmark-read https://example.com/story-a
llm-watch --clear-read
```

Or run as a module:

```bash
python -m llmwatch.main [options]
```

### Read Tracking

Use read tracking to exclude already reviewed stories from future report runs.

- Mark individual URLs: `--mark-read URL ...`
- Mark all links from a report file: `--mark-read-from-report FILE`
- Scope report import to one section: `--section "Common Links"`
- List entries: `--list-read`
- Remove entries: `--unmark-read URL ...`
- Clear all entries: `--clear-read`

Read URLs are normalized before comparison (query strings, fragments, and
trailing slashes are stripped), so tracking remains stable across URL variants.

### Cache Directory

All on-disk caches (read tracker, arXiv lookup, TLDR cache, vendor scrape
health cache) use a shared cache directory.

- Default directory: `.llmwatch_cache` (relative to current working directory)
- Override with env var: `LLMWATCH_CACHE_DIR=/absolute/path`

Examples:

```bash
# Keep all llm-watch cache files in one stable location
export LLMWATCH_CACHE_DIR=/home/me/.cache/llm-watch
llm-watch
```

### arXiv Lookup Caching

The arXiv lookup agent uses a local cache by default to reduce timeouts and
network calls.

- Cache file: `.llmwatch_cache/arxiv_lookup_cache.json`
- Default behavior: look up each normalized query term in cache first
- Override behavior: use `--arxiv-force-fetch` to bypass cache and fetch all
   terms directly from arXiv

If `LLMWATCH_CACHE_DIR` is set, the cache file location becomes
`$LLMWATCH_CACHE_DIR/arxiv_lookup_cache.json`.

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

If `LLMWATCH_CACHE_DIR` is set, the cache file location becomes
`$LLMWATCH_CACHE_DIR/tldr_items.json`.

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

### Ollama Editor Pass (Phase 4)

The `editor` agent post-processes the finished Markdown report using a local
Ollama model.  It is **disabled by default** and must be opted into with
`--edit` or `LLMWATCH_EDITOR_ENABLED=true`.

Five independent editing tasks are available:

| Task | CLI skip key | Default | Description |
|------|--------------|---------|-------------|
| Summary paragraph | `summary` | enabled | Injects a 4–6 sentence narrative digest below the datestamp line |
| Fix truncations | `truncations` | enabled | Repairs TLDR entries cut off before a backtick tag |
| Stale annotations | `stale` | enabled | Adds `<!-- stale: YYYY-MM-DD -->` to items older than 30 days |
| Theme tags | `themes` | disabled | Adds `<!-- theme: slug -->` after each section heading |
| Model digest | `model_digest` | disabled | Replaces Ollama/HuggingFace model-list sections with prose |

Environment knobs:

- `LLMWATCH_EDITOR_ENABLED` (default: `false`) — master switch
- `LLMWATCH_EDITOR_MODEL` (default: `laguna-xs.2`) — Ollama model tag
- `LLMWATCH_EDITOR_BASE_URL` (default: `http://localhost:11434`) — Ollama server
- `LLMWATCH_EDITOR_TIMEOUT` (default: `120`) — HTTP timeout in seconds
- `LLMWATCH_EDITOR_SUMMARY` (default: `true`)
- `LLMWATCH_EDITOR_FIX_TRUNCATIONS` (default: `true`)
- `LLMWATCH_EDITOR_ANNOTATE_STALE` (default: `true`)
- `LLMWATCH_EDITOR_THEME_TAGS` (default: `false`)
- `LLMWATCH_EDITOR_MODEL_DIGEST` (default: `false`)

When Ollama is unavailable the editor result carries an error and the unedited
reporter Markdown is written to disk unchanged.

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
