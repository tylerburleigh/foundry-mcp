# semantic-scholar-enhancements

**Mission**: Enhance the Semantic Scholar provider with extended API fields and filtering options to improve academic paper search quality.

## Objective

Leverage untapped Semantic Scholar API capabilities by adding TLDR summaries, venue information, influential citation counts, publication type filtering, and sorting options to the existing search implementation.

## Scope

### In Scope
- Add extended fields to API requests (tldr, venue, influentialCitationCount, referenceCount, fieldsOfStudy)
- Add publication type filtering (Review, JournalArticle, Conference, etc.)
- Add sorting options (by citationCount, publicationDate, paperId)
- Add parameter validation following Perplexity provider patterns
- Update response parsing to extract and use new fields
- Use TLDR for snippet when available (better than truncated abstract)
- Comprehensive test coverage
- Backward compatibility (all changes opt-out, not opt-in)

### Out of Scope
- Recommendations API (separate endpoint - future phase)
- Paper details lookup by ID (future phase)
- Citation/reference graph traversal (future phase)
- Author search capabilities (future phase)
- SPECTER2 embeddings support (future phase)

## API Contract Reference

Based on Semantic Scholar API documentation (https://api.semanticscholar.org/api-docs/graph):

### Extended Fields (verified)
```
tldr                      - AI-generated summary object {model, text}
influentialCitationCount  - Count of influential citations
referenceCount            - Number of references
venue                     - Publication venue string
fieldsOfStudy             - Array of field objects [{category, source}]
```

### Publication Types Filter (verified)
Valid values for `publicationTypes` parameter:
```
Review, JournalArticle, Conference, CaseReport, ClinicalTrial,
Dataset, Editorial, LettersAndComments, MetaAnalysis, News, Study, Book, BookSection
```
Format: Comma-separated string (e.g., "Review,JournalArticle")

### Sort Options (verified)
- `sort` parameter format: `field:direction`
- Valid fields: `paperId`, `publicationDate`, `citationCount`
- Valid directions: `asc`, `desc` (default: `desc` if omitted)
- Examples: `citationCount:desc`, `publicationDate:asc`

## Parameter Specifications

### Input Formats
| Parameter | Python Type | Serialization | Default |
|-----------|-------------|---------------|---------|
| `publication_types` | `list[str]` | Comma-join → `publicationTypes` | `None` (no filter) |
| `sort_by` | `str` | Combined with sort_order → `sort` | `None` (relevance) |
| `sort_order` | `str` | `"asc"` or `"desc"` | `"desc"` |
| `use_extended_fields` | `bool` | Selects field set | `True` |

### Precedence Rules
1. `use_extended_fields=True` → Use `EXTENDED_FIELDS` constant
2. `use_extended_fields=False` → Use `DEFAULT_FIELDS` constant
3. No existing `fields` parameter override exists - these are the only options
4. Filtering (`publication_types`) and sorting (`sort_by`) work independently of field selection

### Validation Rules
- `publication_types`: Case-sensitive match against `VALID_PUBLICATION_TYPES`
- `sort_by`: Must be in `VALID_SORT_FIELDS` or `None`
- `sort_order`: Must be `"asc"` or `"desc"` (only validated if `sort_by` is set)
- If `sort_order` provided without `sort_by`, ignore `sort_order` (no error)

### Snippet Fallback Chain
1. If `tldr.text` exists and non-empty → use as snippet
2. Else if `abstract` exists → truncate to 500 chars at word boundary + "..."
3. Else → `None`

## Phases

### Phase 1: Add Extended Fields and Constants

**Purpose**: Define new API field constants and validation rules

**Tasks**:
1. Add `EXTENDED_FIELDS` constant with new fields (tldr, influentialCitationCount, referenceCount, venue, fieldsOfStudy)
2. Add `VALID_PUBLICATION_TYPES` frozenset with valid filter values (case-sensitive)
3. Add `VALID_SORT_FIELDS` frozenset with valid sort options
4. Add `_validate_search_params()` function following Perplexity pattern

**Verification**:
- Constants defined correctly
- Validation function raises ValueError for invalid inputs with clear messages
- Fidelity review: Constants match API documentation

### Phase 2: Update Search Method

**Purpose**: Accept and process new search parameters

**Tasks**:
1. Extract new kwargs: publication_types, sort_by, sort_order, use_extended_fields
2. Call validation function with new parameters
3. Build params dict with extended fields (conditional on use_extended_fields)
4. Add publicationTypes filter to params when provided (comma-joined)
5. Add sort parameter to params when provided (format: "field:direction")
6. Update docstring with new kwargs documentation

**Verification**:
- New parameters accepted without breaking existing calls
- API requests include correct params with proper formatting
- Fidelity review: Parameter handling matches spec

### Phase 3: Update Response Parsing

**Purpose**: Extract and use new fields from API response

**Tasks**:
1. Extract TLDR object and text from response (handle missing/null gracefully)
2. Use TLDR text for snippet when available, fallback to truncated abstract
3. Add venue to metadata (may be null)
4. Add influential_citation_count to metadata (may be null)
5. Add reference_count to metadata (may be null)
6. Add fields_of_study to metadata (may be empty array)
7. Add tldr to metadata for direct access

**Verification**:
- New fields appear in ResearchSource metadata
- TLDR used as snippet when available
- Missing fields handled gracefully (no errors)
- Fidelity review: Response parsing matches spec

### Phase 4: Test Coverage

**Purpose**: Ensure reliability and catch regressions

**Tasks**:
1. Create test_semantic_scholar.py test file
2. Create semantic_scholar_responses.py fixtures file with:
   - Basic response fixture
   - Response with TLDR fixture
   - Response with all extended fields
   - Response with missing/null fields
   - Error response fixtures (429 rate limit)
3. Add initialization tests (API key handling)
4. Add validation tests:
   - Invalid publication_types raises ValueError
   - Invalid sort_by raises ValueError
   - Invalid sort_order raises ValueError
   - sort_order without sort_by is ignored (no error)
5. Add extended fields parsing tests:
   - TLDR extraction populates snippet
   - Missing TLDR falls back to abstract
   - All metadata fields extracted correctly
   - Missing fields default to None/empty
6. Add parameter building tests:
   - publicationTypes joins correctly
   - sort parameter formats correctly
   - use_extended_fields=True uses EXTENDED_FIELDS
   - use_extended_fields=False uses DEFAULT_FIELDS
7. Add backward compatibility test (no new params = same behavior)

**Verification**:
- All tests pass
- Coverage >90% for new code paths
- Fidelity review: Test coverage matches spec requirements

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Extended fields increase response size | Low | Default True, but can disable; measure in testing |
| TLDR not available for all papers | Low | Graceful fallback to truncated abstract |
| Publication types change over time | Medium | Clear error messages; easy to update frozenset |
| Sort parameter format changes | Low | Simple format, well-documented |
| Breaking existing integrations | High | All new params have defaults, extensive backward compat tests |
| API fields don't match documentation | Medium | Verify with real API calls in manual testing |

## Success Criteria

- [ ] Extended fields (tldr, venue, influentialCitationCount, referenceCount, fieldsOfStudy) returned in metadata
- [ ] TLDR used as snippet when available, with correct fallback chain
- [ ] publication_types filter works correctly (comma-joined)
- [ ] sort_by and sort_order work correctly (field:direction format)
- [ ] Invalid parameters raise clear ValueError messages
- [ ] use_extended_fields=False falls back to DEFAULT_FIELDS
- [ ] Missing/null fields handled gracefully
- [ ] All existing deep_research tests continue to pass
- [ ] New test file with >90% coverage of new code

## Files to Modify

| File | Changes |
|------|---------|
| `src/foundry_mcp/core/research/providers/semantic_scholar.py` | Extended fields, validation, new params, parsing |
| `tests/core/research/providers/test_semantic_scholar.py` | Create new test file |
| `tests/fixtures/semantic_scholar_responses.py` | Create new fixtures file |

## Dependencies

- Perplexity provider pattern reference: `src/foundry_mcp/core/research/providers/perplexity.py` (lines 89-150 for validation)
- Base provider interface: `src/foundry_mcp/core/research/providers/base.py`
- Existing Semantic Scholar implementation: `src/foundry_mcp/core/research/providers/semantic_scholar.py`
