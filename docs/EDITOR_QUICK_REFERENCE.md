# Quick Reference: Ollama Editor Pass (Active)

Status: Active and current as of 2026-05-04

This is the canonical editor reference for current behavior and deferred follow-up work.

## Current Implementation

The editor runs as an optional Phase 4 after the reporter has produced the final
Markdown report and before the file is written to disk.

### Implemented Files

| File | Status |
|------|--------|
| `llmwatch/agents/editor.py` | Implemented |
| `llmwatch/ollama_client.py` | Implemented |
| `llmwatch/orchestrator.py` | Updated for optional editor phase |
| `llmwatch/main.py` | Updated with editor CLI flags |
| `tests/test_editor.py` | Implemented |
| `tests/test_orchestrator.py` | Updated with editor-phase coverage |

### Current Behavior

- Disabled by default; enabled with `--edit` or `LLMWATCH_EDITOR_ENABLED=true`
- Uses a local Ollama model through `POST /api/chat`
- Runs tasks independently so one task failure does not abort the whole editor pass
- Falls back to the unedited reporter output when Ollama is unavailable or the editor produces no replacement report body

### Implemented Tasks

| Task | Default | Purpose |
|------|---------|---------|
| `summary` | enabled | Insert a short executive summary near the top of the report |
| `truncations` | enabled | Repair list items that appear cut off mid-sentence |
| `stale` | enabled | Annotate items older than 30 days |
| `themes` | disabled | Add `<!-- theme: ... -->` tags to headings |
| `model_digest` | disabled | Replace model-list sections with prose digests |

## Deferred Follow-Up Work

These items are intentionally recorded for later. They are not part of the current implementation task.

### Priority 1: Reliability

1. Harden editor tasks against reporter-format drift
   - Summary insertion currently depends on the `*Generated: ...*` marker.
   - Truncation and model-digest behavior rely on regexes that may silently no-op if report formatting changes.
2. Add golden-file tests using real weekly report fixtures
   - Validate full report editing end to end.
   - Cover both enabled and fallback paths against realistic report shapes.
3. Tighten idempotency guarantees
   - Ensure running the editor twice does not duplicate or compound changes.
   - Mechanical tasks already behave better here than LLM-based tasks.

### Priority 2: Operability

1. Add edit visibility and diff support
   - Show what changed per task.
   - Make it easy to compare edited and original output.
2. Add finer CLI controls
   - Per-task enable flags.
   - An editor-only dry run.
   - Optional write-both mode for original and edited reports.
3. Improve task-level observability
   - Record which tasks ran, changed content, no-op'd, or failed.
   - Capture simple counts so the editor can be evaluated over time.

### Priority 3: Product Decisions

1. Decide whether `model_digest` should replace lists or preserve structure
   - Prose is easier to read.
   - Raw lists are easier to inspect and may matter for downstream parsing.
2. Define the editor's allowed scope of mutation
   - Determine whether the editor may only polish wording or also reshape report structure.
3. Add optional live Ollama integration coverage
   - Keep unit tests as the default.
   - Add a smoke test path for local environments with Ollama available.

## Suggested Next Steps When This Work Resumes

1. Add golden tests using one or two real report snapshots.
2. Add a diff-oriented CLI option such as `--editor-show-diff`.
3. Harden `summary`, `truncations`, and `model_digest` against minor reporter format changes.