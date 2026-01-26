# Perplexity API Enhancements

## Mission

Enhance the Perplexity search provider (`/search` endpoint) with additional configurable parameters, following patterns established by the Tavily provider enhancement.

## Objective

Expose full Perplexity Search API capabilities by implementing missing search parameters, adding configuration fields to ResearchConfig, and integrating with the deep research workflow.

## Scope

### In Scope
- Add `search_context_size` parameter (low/medium/high) for cost/quality control
- Make `max_tokens` and `max_tokens_per_page` configurable (currently hard-coded)
- Add date range filters (`search_after_date`, `search_before_date`)
- Document domain filter "-" prefix for exclusion (already supported by API)
- Add 5 Perplexity configuration fields to ResearchConfig
- Update TOML parsing for new config fields
- Pass Perplexity kwargs from deep research workflow
- Unit tests for new parameters and config fields
- Parameter validation following Tavily patterns

### Out of Scope
- Chat Completions API (`/chat/completions`) - future enhancement
- Model selection (sonar-pro, sonar-deep-research, sonar-reasoning-pro)
- `search_mode` (academic, sec, web) - requires Chat Completions API
- `return_images`, `return_related_questions` - requires Chat Completions API
- `user_location` for geo-relevance - requires Chat Completions API
- Streaming responses
- Structured JSON output

## Background Research

### Perplexity Search API Reference

**Source**: https://docs.perplexity.ai/

**Current Implementation** (`perplexity.py`):

| Parameter | Supported | Notes |
|-----------|-----------|-------|
| `query` | Yes | Required |
| `max_results` | Yes | 1-20, clamped |
| `search_recency_filter` | Yes | day, week, month, year |
| `search_domain_filter` | Yes | Max 20 domains, supports "-" prefix for exclusion |
| `country` | Yes | Geographic filter |
| `max_tokens` | Hard-coded | 50000 |
| `max_tokens_per_page` | Hard-coded | 2048 |

**Missing Parameters** (from API docs - verified at https://docs.perplexity.ai/guides/search-date-time-filters):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_context_size` | string | "medium" | "low", "medium", "high" - affects cost |
| `search_after_date` | string | None | Filter after publication date (%m/%d/%Y) |
| `search_before_date` | string | None | Filter before publication date (%m/%d/%Y) |
| `last_updated_after_filter` | string | None | Filter by last modified date (%m/%d/%Y) |
| `last_updated_before_filter` | string | None | Filter by last modified date (%m/%d/%Y) |

**Note**: `search_recency_filter` cannot be combined with specific date filters - must choose one approach.

**Pricing by Context Size**:

| search_context_size | Cost per 1K requests |
|---------------------|---------------------|
| low | $5 |
| medium | $8 |
| high | $12 |

### Current Implementation Files

- `src/foundry_mcp/core/research/providers/perplexity.py` - Search provider (450 lines)
- `src/foundry_mcp/config.py` - ResearchConfig class
- `src/foundry_mcp/core/research/workflows/deep_research.py` - Consumer
- `tests/core/research/providers/test_perplexity.py` - Existing tests (497 lines)

### Key Finding

The deep research workflow never passes Perplexity-specific kwargs - it always uses defaults. The `max_tokens` and `max_tokens_per_page` are hard-coded, preventing users from controlling response size and cost.

## Phases

### Phase 1: Enhanced Search Parameters

**Purpose**: Add configurable parameters to existing provider

**Tasks**:
1. Add `VALID_SEARCH_CONTEXT_SIZES` constant (frozenset)
2. Extract new parameters in search method:
   - `search_context_size` (default: "medium")
   - `max_tokens` (default: 50000, now configurable)
   - `max_tokens_per_page` (default: 2048, now configurable)
   - `search_after_date` (format: %m/%d/%Y) - publication date filter
   - `search_before_date` (format: %m/%d/%Y) - publication date filter
3. Update payload construction to include new parameters
4. Update docstrings to document "-" prefix support for domain exclusion (already supported by API)
5. Add `_validate_search_params()` function following Tavily pattern
6. Update docstrings with new parameters

**Files**:
- `src/foundry_mcp/core/research/providers/perplexity.py`

**Verification**:
- Unit tests pass for each new parameter
- Payload correctly includes parameters when set
- Invalid values raise ValueError with clear message

### Phase 2: Configuration Support

**Purpose**: Add configurable Perplexity settings to ResearchConfig

**Tasks**:
1. Add Perplexity config fields to `ResearchConfig` dataclass (following `tavily_` prefix pattern):
   - `perplexity_search_context_size: str = "medium"`
   - `perplexity_max_tokens: int = 50000`
   - `perplexity_max_tokens_per_page: int = 2048`
   - `perplexity_recency_filter: Optional[str] = None`
   - `perplexity_country: Optional[str] = None`
2. Update `from_toml_dict()` to parse new fields
3. Add validation in `__post_init__()`:
   - `search_context_size`: must be "low", "medium", or "high"
   - `max_tokens`: must be positive integer
   - `max_tokens_per_page`: must be positive integer
   - `recency_filter`: must be day, week, month, year, or None

**Files**:
- `src/foundry_mcp/config.py`

**Verification**:
- Config loads from TOML correctly
- Default values applied when not specified
- Invalid values raise validation errors with clear messages

### Phase 3: Deep Research Integration

**Purpose**: Pass Perplexity config to the search provider

**Context**: The workflow already has `_get_tavily_search_kwargs()` (line 1060) which is called at line 2753 when `provider_name == "tavily"`. The non-Tavily branch (line 2755-2757) currently only sets `include_raw_content`. We need to add Perplexity-specific handling.

**Tasks**:
1. Add `_get_perplexity_search_kwargs(state: DeepResearchState) -> dict[str, Any]` helper method following the Tavily pattern
2. Update the provider selection logic (around line 2755) to call `_get_perplexity_search_kwargs()` when `provider_name == "perplexity"`
3. Read config via `self.config.research` (ResearchConfig instance) to access `perplexity_*` fields
4. Document parameter precedence: explicit kwargs > config values > defaults

**Files**:
- `src/foundry_mcp/core/research/workflows/deep_research.py`

**Verification**:
- Config values are passed to Perplexity provider
- Explicit kwargs override config values
- Default behavior preserved when no config set

### Phase 4: Testing

**Purpose**: Ensure comprehensive test coverage for all changes

**Tasks**:
1. Create `tests/unit/test_config_perplexity.py` for config tests:
   - Test TOML parsing for all new fields
   - Test default value preservation
   - Test validation errors for invalid values
   - Test precedence rules
2. Extend `tests/core/research/providers/test_perplexity.py`:
   - Test search_context_size parameter in payload
   - Test max_tokens override
   - Test date filter parameters
   - Test domain deny-list with "-" prefix
   - Test validation error messages
3. Create `tests/fixtures/perplexity_responses.py`:
   - Mock responses for new parameter scenarios
   - Follow pattern from `tests/fixtures/tavily_responses.py`

**Files**:
- `tests/unit/test_config_perplexity.py` (NEW)
- `tests/core/research/providers/test_perplexity.py`
- `tests/fixtures/perplexity_responses.py` (NEW)

**Verification**:
- All tests pass: `pytest tests/core/research/providers/test_perplexity.py -v`
- Config tests pass: `pytest tests/unit/test_config_perplexity.py -v`
- Coverage includes new code paths

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| API costs increase with "high" context | Medium | Default to "medium"; document cost implications |
| Breaking existing behavior | High | All new parameters optional with backward-compatible defaults |
| Perplexity API changes | Low | Use kwargs pattern for forward compatibility |
| Date format validation complexity | Low | Use strict %m/%d/%Y format with clear error messages |

## Technical Specifications

### Parameter Validation Rules

```python
VALID_SEARCH_CONTEXT_SIZES = frozenset(["low", "medium", "high"])
VALID_RECENCY_FILTERS = frozenset(["day", "week", "month", "year"])

def _validate_search_params(
    search_context_size: str,
    max_tokens: int | None,
    max_tokens_per_page: int | None,
    search_after_date: str | None,
    search_before_date: str | None,
    recency_filter: str | None,
) -> None:
    """Validate search parameters. Raises ValueError if invalid."""
    if search_context_size not in VALID_SEARCH_CONTEXT_SIZES:
        raise ValueError(f"Invalid search_context_size: {search_context_size}")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive: {max_tokens}")

    # Date format validation: %m/%d/%Y
    for date_val, name in [(search_after_date, "search_after_date"),
                           (search_before_date, "search_before_date")]:
        if date_val:
            try:
                datetime.strptime(date_val, "%m/%d/%Y")
            except ValueError:
                raise ValueError(f"Invalid {name}: {date_val}. Use format MM/DD/YYYY")

    # Date range logic validation
    if search_after_date and search_before_date:
        after = datetime.strptime(search_after_date, "%m/%d/%Y")
        before = datetime.strptime(search_before_date, "%m/%d/%Y")
        if after >= before:
            raise ValueError(
                f"search_after_date ({search_after_date}) must be before "
                f"search_before_date ({search_before_date})"
            )

    # Cannot combine recency_filter with specific dates
    if recency_filter and (search_after_date or search_before_date):
        raise ValueError(
            "Cannot combine recency_filter with search_after_date/search_before_date"
        )
```

### Domain Filter Enhancement

```python
# Current: include-only
domain_filter = ["example.com", "docs.python.org"]

# Enhanced: support "-" prefix for exclusion
domain_filter = ["example.com", "-spam.com", "-ads.example.org"]
```

### Config TOML Example

```toml
[research]
perplexity_search_context_size = "high"
perplexity_max_tokens = 50000
perplexity_max_tokens_per_page = 2048
perplexity_recency_filter = "week"
perplexity_country = "US"
```

## Success Criteria

- [ ] All 5 new search parameters are supported in perplexity.py
- [ ] Configuration fields added to ResearchConfig with TOML support (following `tavily_` prefix pattern)
- [ ] `search_context_size` defaults to "medium" (balanced cost/quality)
- [ ] Deep research workflow passes Perplexity config to provider
- [ ] Domain filter "-" prefix for exclusion documented in docstrings
- [ ] Date range validation: after_date < before_date when both provided
- [ ] Unit tests cover new parameters and config fields
- [ ] All existing tests continue to pass (backward compatibility)
- [ ] Invalid parameter values raise clear ValueError messages

## Dependencies

- `httpx` (already used)
- Perplexity API key in environment (`PERPLEXITY_API_KEY`)
- No new external dependencies required

## Assumptions

1. Perplexity Search API endpoints remain stable
2. Users have valid Perplexity API keys configured
3. Default `search_context_size="medium"` provides reasonable balance
4. Date format %m/%d/%Y is acceptable (matches Perplexity API)
