# Quick Reference: Common Links Consolidation (Active)

Status: Active and current as of 2026-05-02

This is the canonical consolidation reference. Historical planning and rollout docs are archived in [CONSOLIDATION_AGENT_INVESTIGATION.md](CONSOLIDATION_AGENT_INVESTIGATION.md) and [PHASE_1_MVP_IMPLEMENTATION.md](PHASE_1_MVP_IMPLEMENTATION.md).

## Problem Statement
News items appear **2-3 times** across different report sections:
- TLDR AI → "Trending & New Models"
- Last Week in AI → "Referenced Links"  
- The Neuron → "Summaries"
- ArXiv → "Research Papers"

## Solution: Consolidator Agent
A **reporter-phase agent** that:
1. Detects duplicate stories across sources
2. Creates "Common Links" section with cross-references
3. Enriches metadata (impact score, appearance count)

## Current Implementation

### Implemented Files

| File | Status |
|------|--------|
| `llmwatch/agents/consolidator.py` | Implemented |
| `llmwatch/agents/reporter.py` | Updated for Common Links bucketing + suppression summary |
| `llmwatch/orchestrator.py` | Updated for sequential reporter execution |
| `tests/test_consolidator.py` | Implemented (URL/similarity + ranking/suppression coverage) |

### Architecture

```
Watcher Phase (collect data)
    ↓
Lookup Phase (find papers)
    ↓
Reporter Phase (sequential)
    ├─→ StoryConsolidatorAgent
    │       • Input: watcher_results + lookup_results
    │       • Output: consolidated_stories with metadata
    │       • Algorithm: URL normalization + title similarity + weighted ranking + suppression
    ├─→ WeeklyReporterAgent (modified)
    │       • Input: + consolidated_stories
    │       • Output: markdown with "Common Links" section
    └─→ Other reporters
```

## Consolidation + Ranking Algorithm (Current)

**Primary match**: Normalize URLs, exact match
```python
url1 = "https://example.com/article/grok" 
url2 = "https://example.com/article/grok?utm_source=tldr"
→ Both normalized to "https://example.com/article/grok" ✓ MATCH
```

**Secondary match**: Title similarity (>85%) + same week
```python
title1 = "xAI launches Grok 4.3"
title2 = "Grok 4.3 improves cost-efficiency"
→ Similarity: 0.86 ✓ MATCH (if within 7 days)
```

Current behavior is URL + title similarity within temporal window, followed by weighted ranking and suppression filtering.

### Ranking Signal (Current)

`common_link_signal` combines:

1. Source-class diversity (newsletter/podcast/model hub/research hub)
2. Source diversity and appearance count
3. Freshness score (recent links boosted, stale links penalized)
4. Novelty score (newly surfaced links lightly boosted, evergreen wide-span links penalized)

### Suppression Rules (Current)

Suppression runs after consolidation and classification:

- Suppress sponsor links by default
- Suppress single-source social links by default
- Suppress configured domains
- Allowlist domains override all suppression rules

## Output Example

### Common Links Section (New)
```markdown
## Common Links This Week

### 1. xAI Launches Grok 4.3 – Improved Cost-Efficiency 🔥
**Impact**: Covered by 3 sources | **First reported**: May 2, 2026

Grok 4.3 scores higher on Intelligence Index while costing less...

**Coverage**:
- TLDR AI Daily Newsletter (May 2 | Headlines & Launches)
- Last Week in AI Podcast #242 (Apr 30 | Referenced)
- The Neuron Newsletter (May 1)

---
```

## Phase Progress

### Phase 1: MVP (URL-based deduplication)
Status: Completed | Coverage: ~70% of duplicates

```python
def consolidate_by_url(items):
    """Simple URL-based deduplication."""
    seen = {}
    for item in items:
        url = normalize_url(item.get("url", ""))
        if not url or url not in seen:
            seen[url] = item
            yield item
```

### Phase 2: Enhanced (Title similarity)
Status: Completed | Coverage: ~90% of duplicates

```python
def consolidate_with_similarity(items):
    """URL match + title similarity (0.85 threshold)."""
    # Requires: difflib.SequenceMatcher
    # Adds: temporal proximity check (same week)
```

### Phase 3: Quality Ranking (Implemented)
Status: Completed

```python
class StoryConsolidatorAgent(BaseAgent):
    """Current implementation: URL + title similarity + weighted common-link ranking."""
    # Implemented:
    # - URL normalization and grouping
    # - SequenceMatcher title similarity (threshold configurable)
    # - Temporal window filtering
    # - Cross-source appearances metadata
    # - Source-class weighted scoring
    # - Freshness/novelty adjustments
    # - Suppression rules with allowlist override
```

## Key Decision Points

| Decision | Current State | Rationale |
|----------|---------------|----------|
| **Create new agent or in-reporter?** | Implemented as new agent | Cleaner separation of concerns; testable |
| **Similarity threshold** | 0.85 default, configurable | Reduces false positives; tunable via env var |
| **Temporal window** | 7 days default, configurable | Stories considered same when temporally close |
| **Ranking metric** | Implemented (weighted signal) | Prioritizes independent cross-source agreement + recency |
| **Suppression policy** | Implemented (configurable) | Reduces low-signal recurring links while preserving visibility in summary |
| **Future: Local AI model** | Planned | Optional semantic deduplication for hard title mismatches |

### Configuration
```python
# Environment variables for consolidator
LLMWATCH_CONSOLIDATOR_SIMILARITY_THRESHOLD=0.85  # Configurable 0.0-1.0
LLMWATCH_CONSOLIDATOR_TEMPORAL_WINDOW_DAYS=7     # Temporal proximity window
LLMWATCH_CONSOLIDATOR_SUPPRESS_SPONSORS=true
LLMWATCH_CONSOLIDATOR_SUPPRESS_SOCIAL_SINGLE_SOURCE=true
LLMWATCH_CONSOLIDATOR_SUPPRESS_DOMAINS=example.com,another.com
LLMWATCH_CONSOLIDATOR_ALLOW_DOMAINS=trusted.example.com
LLMWATCH_CONSOLIDATOR_OLLAMA_ENABLED=false       # Future: Enable local ML dedup
LLMWATCH_CONSOLIDATOR_OLLAMA_MODEL=llama3.2:3b   # Future: Local model for complex cases
```

## Testing Strategy

```python
# Unit tests (implemented):
test_normalize_url()           # URL normalization
test_title_similarity()        # Similarity matching
test_consolidate_basic()       # Simple dedup
test_consolidate_complex()     # Multi-source same story
test_common_links_render() # Markdown generation
test_suppression_rules()       # Sponsor/domain/social suppression + allowlist
test_freshness_novelty()       # Recency and novelty weighting

# Integration tests (implemented):
test_end_to_end_with_sample_report()  # Full flow
test_consolidator_ordering()          # Orchestrator sequencing
```

## Known Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **False positives** (different stories match) | Increase similarity threshold to 0.90 or require URL match |
| **Performance** (O(n²) matching) | Pre-filter by date before matching; use bloom filters for URLs |
| **Missing URLs** | Use description hash as fallback matching |
| **Different titles, same story** | Use description embedding similarity (requires ML model) |
| **Changing requirement** | Consolidator is isolated; easy to modify/extend |

## Observed Metrics

Recent run metrics:
- Phase 1 baseline: 189 items -> 187 unique stories
- Phase 2 current: 189 items -> 181 unique stories
- Additional Phase 2 gain: 6 more duplicates consolidated
- Common Links section rendered in report output

## Roadmap

### Phase 1: MVP – URL-Based Deduplication
Status: Completed

**Deliverables**:
- ✓ Create `agents/consolidator.py` with `StoryConsolidatorAgent`
- ✓ URL normalization & exact matching
- ✓ Configuration support (similarity threshold, temporal window)
- ✓ Basic "Common Links" section rendering
- ✓ Unit tests for URL normalization
- ✓ Integration test with sample report

**Definition of Done**:
- Duplicates by URL are eliminated
- "Common Links" section appears in report
- No regression in existing report sections
- Configuration can be set via environment variables
- Tests passing

### Phase 2: Enhanced – Title Similarity
Status: Completed

**Additions**:
- Title similarity matching (configurable threshold: 0.85, default)
- Temporal window filtering (7 days, configurable)
- `SequenceMatcher`-based fuzzy matching
- Enhanced deduplication tests

### Phase 3: Future – Local AI Model Support
Status: Backlog

**Approach** (similar to TLDR AI's Ollama filtering):
- Integrate Ollama for semantic understanding when enabled
- Use local model for ambiguous matches
- Example prompt:
  ```
  "Are these two stories about the same event?
   Story 1: {title1} - {desc1}
   Story 2: {title2} - {desc2}
   Return JSON: {same: boolean, confidence: 0.0-1.0}"
  ```
- Environment variables: `LLMWATCH_CONSOLIDATOR_OLLAMA_ENABLED`, `LLMWATCH_CONSOLIDATOR_OLLAMA_MODEL`
- Graceful fallback if Ollama unavailable (use Phase 2 matching)

## Immediate Next Steps

1. Add optional domain-pattern suppression (regex path-level rules)
2. Add report-level counts for suppressed reasons (sponsor/domain/social)
3. Prototype optional semantic deduplication with Ollama for ambiguous title pairs
