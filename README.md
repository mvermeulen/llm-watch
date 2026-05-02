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
| `watcher`  | HuggingFace, Ollama | Fetch trending/new models from public sources             |
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
```

Or run as a module:

```bash
python -m llmwatch.main [options]
```

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
