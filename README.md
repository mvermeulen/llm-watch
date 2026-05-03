# llm-watch

> Keep track and report on large language models.

A modular, agent-based system that monitors the world of large language models
and produces a **weekly investigation report** in Markdown.

## Architecture

```
llmwatch/
├── agents/
│   ├── base.py            # BaseAgent, AgentResult, AgentRegistry (registry singleton)
│   ├── watchers/
│   │   ├── huggingface.py # Watches HuggingFace trending models
│   │   ├── neuron_feed.py # Watches The Neuron Atom feed (feed-first mode)
│   │   └── ollama.py      # Watches the Ollama model library
│   ├── lookup/
│   │   └── arxiv.py       # Looks up arXiv papers for discovered models
│   └── reporter.py        # Aggregates findings into a weekly Markdown report
├── orchestrator.py        # Runs agents in three phases: watch → lookup → report
└── main.py                # CLI entry point
```

### Agent phases

| Phase      | Agents           | Description                                                  |
|------------|------------------|--------------------------------------------------------------|
| `watcher`  | HuggingFace, Ollama, The Neuron | Fetch trending/new models from public sources |
| `lookup`   | arXiv            | Search for papers related to discovered models               |
| `reporter` | WeeklyReporter   | Aggregate everything into a Markdown report                  |

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

# Fetch TLDR data for a specific date range and merge with cache
llm-watch --tldr-fetch-only --tldr-date-range "2026-04-25:2026-05-02"

# Set Last Week in AI podcast lookback window (days)
llm-watch --lwiai-lookback-days 14

# Set The Neuron feed lookback window (days)
llm-watch --neuron-lookback-days 7
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
four sections:

1. **Trending & New Models** – models discovered by watcher agents, grouped by
   source.
2. **Related Research Papers (arXiv)** – papers found by the lookup agent,
   grouped by search term.
3. **Warnings** – any errors that occurred during the run.
4. **New Sources Discovered** – external URLs found in model descriptions or
   paper abstracts that are not yet tracked, highlighted for future monitoring.
