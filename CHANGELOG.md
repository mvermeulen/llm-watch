# Changelog

All notable changes to this project are documented in this file.

## 2026-05-05

- Added read-tracking CLI workflows:
  - `--mark-read URL ...`
  - `--unmark-read URL ...`
  - `--list-read`
  - `--clear-read`
  - `--mark-read-from-report FILE`
  - `--section "..."` (scope report import to one H2 section)
- Added report-link import flow so URLs can be marked read directly from existing Markdown reports.
- Read URLs now suppress matching stories in subsequent reports (with URL normalization).
- Added configurable cache root via `LLMWATCH_CACHE_DIR`.
- Unified cache location handling across read tracker, TLDR cache, arXiv lookup cache, and vendor scrape health cache.
