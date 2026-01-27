# Document Digest for Deep Research

## Mission

Add an internal, provider-agnostic document digest step for deep research that compresses large HTML/PDF content into structured JSON summaries with evidence snippets, preventing context budget overflow.

## Objective

Insert a digest phase into the deep research pipeline that:
1. Identifies sources eligible for compression (large content >10k chars, PDFs, high-quality sources)
2. Extracts and compresses content using the existing ContentSummarizer infrastructure
3. Preserves evidence snippets for citation support
4. Records fidelity metadata for transparency
5. Archives raw content when enabled

## Scope

### In Scope
- New `DocumentDigestor` class in `src/foundry_mcp/core/research/document_digest.py`
- Versioned `DigestPayload` schema with explicit content-type signaling
- PDF text extraction using pypdf (with pdfminer.six as optional fallback)
- Integration with existing `ContentSummarizer` map-reduce pipeline
- New `FidelityLevel.DIGEST` enum value for fidelity tracking
- Evidence snippet contract with citation traceability
- Configuration fields in `ResearchConfig` for digest behavior
- Audit events and metrics for observability
- Unit and integration tests

### Out of Scope
- No new MCP tool or CLI surface (internal-only)
- No changes to existing summarization levels (CONDENSED, KEY_POINTS, etc.)
- No streaming/real-time digest updates
- No external caching layer (uses existing SummaryCache)

---

## Digest Contract Specification

### Design Decision: Query-Conditioned Digests

**Decision**: Digests ARE query-conditioned. The summary focus and evidence selection depend on the research query.

**Rationale**:
- Evidence snippets are selected based on relevance to the query
- Summary may emphasize query-relevant aspects
- This improves citation quality for the specific research task

**Implication**: Cache keys MUST include `query_hash`.

### On-Wire JSON Schema (DigestPayload v1.0)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "content_type": { "type": "string", "const": "digest/v1" },
    "query_hash": { "type": "string", "pattern": "^[a-f0-9]{8}$" },
    "summary": { "type": "string", "maxLength": 2000 },
    "key_points": {
      "type": "array",
      "items": { "type": "string", "maxLength": 500 },
      "maxItems": 10
    },
    "evidence_snippets": {
      "type": "array",
      "items": { "$ref": "#/definitions/EvidenceSnippet" },
      "maxItems": 10
    },
    "original_chars": { "type": "integer", "minimum": 0 },
    "digest_chars": { "type": "integer", "minimum": 0 },
    "compression_ratio": { "type": "number", "minimum": 0, "maximum": 1 },
    "source_text_hash": { "type": "string", "pattern": "^sha256:[a-f0-9]{64}$" }
  },
  "required": ["version", "content_type", "query_hash", "summary", "key_points", "evidence_snippets", "original_chars", "digest_chars", "compression_ratio", "source_text_hash"],
  "definitions": {
    "EvidenceSnippet": {
      "type": "object",
      "properties": {
        "text": { "type": "string", "maxLength": 500 },
        "locator": { "type": "string" },
        "relevance_score": { "type": "number", "minimum": 0, "maximum": 1 }
      },
      "required": ["text", "locator", "relevance_score"]
    }
  }
}
```

**Self-Describing Fields**: The payload includes `content_type` and `query_hash` for portability - consumers can identify and validate the digest even without surrounding metadata.

### Evidence Locator Formats (Per Content Type)

| Content Type | Locator Format | Example | Notes |
|--------------|----------------|---------|-------|
| HTML/Text | `char:{start}-{end}` | `char:1500-1800` | Character offsets in canonical text |
| PDF | `page:{n}:char:{start}-{end}` | `page:3:char:200-450` | Page number (1-based) + offset within page |
| PDF (no page) | `char:{start}-{end}` | `char:5000-5300` | Fallback if page detection fails |

### Locator Indexing Semantics

| Property | Value | Example |
|----------|-------|---------|
| **Index base** | 0-based | First char is position 0 |
| **End boundary** | Exclusive | `char:10-20` = chars at positions 10-19 |
| **Page numbers** | 1-based | `page:1` = first page |
| **Reference text** | Canonical text | Offsets computed against normalized text |

**Slice equivalence**: `locator = "char:10-20"` → `canonical_text[10:20]` in Python

**No text mutation**: Evidence snippets store the exact substring from canonical text. Display truncation (e.g., `...`) is applied at rendering time, NOT stored in the snippet. This preserves offset validity.

**Traceability Guarantee**: When archival is enabled, `source_text_hash` matches the hash of the archived canonical text. Locators reference offsets in the archived text, enabling verification via `archived_text[start:end] == snippet.text`.

### Consumer Branching Rules

1. **Detection**: Check `source.content_type == "digest/v1"` (or `source.is_digest == True`)
2. **Parsing**: Deserialize `source.content` as JSON, validate against schema
3. **Downstream summarization**: SKIP further summarization for digest sources (already compressed)
4. **Citation generation**: Use `evidence_snippets[].text` for quotes, `locator` for attribution
5. **Budget allocation**: Use `digest_chars` (not `original_chars`) for token estimation

### Storage Location

- `source.content`: Serialized DigestPayload JSON string
- `source.content_type`: `"digest/v1"` (new field, default `"text/plain"`)
- `source.metadata["_digest_archive_hash"]`: Archive hash when archival enabled

---

## Archival Contract

### When Archival is Enabled

Controlled by existing config: `deep_research_archive_content = true`

### Storage Specification

| Property | Value |
|----------|-------|
| **Storage path** | `{archive_dir}/{source_id}/{source_text_hash}.txt` |
| **Archive dir** | `~/.foundry-mcp/research_archives/` (default) |
| **Format** | UTF-8 encoded canonical text (post-normalization) |
| **Retention** | 30 days (configurable via `deep_research_archive_retention_days`) |

### Archival Sequence

```
1. Canonicalize raw content → canonical_text
2. Compute source_text_hash = SHA256(canonical_text.encode('utf-8'))
3. Write canonical_text to {archive_dir}/{source_id}/{source_text_hash}.txt
4. Store hash in DigestPayload.source_text_hash
5. Proceed with digest (chunking, evidence extraction uses canonical_text)
```

### Verification Contract

When archival is enabled, this invariant MUST hold:
```python
archived_content = read_file(f"{archive_dir}/{source_id}/{payload.source_text_hash}.txt")
assert hashlib.sha256(archived_content.encode()).hexdigest() == payload.source_text_hash.split(':')[1]

for snippet in payload.evidence_snippets:
    start, end = parse_locator(snippet.locator)  # Returns (int, int)
    assert archived_content[start:end] == snippet.text
```

### When Archival is Disabled

- `source_text_hash` is still computed (for cache keys and fidelity tracking)
- No file written to disk
- Locator verification not possible after digest completes
- `_digest_archive_hash` metadata NOT set

---

## Raw Content Lifecycle

### Lifecycle Definition

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. GATHER      2. RANK           3. DIGEST         4. CLEANUP          │
│  ───────────    ──────────        ─────────         ────────            │
│  source.content → compute features → _raw_content    → DELETE _raw_content│
│  (raw text)      (quality, relevance) (temp copy)     source.content=digest│
└─────────────────────────────────────────────────────────────────────────┘
```

### Explicit Rules

1. **Creation**: `_raw_content` is set ONLY during digest phase, copying from `source.content`
2. **Usage**: Used for ranking feature computation and evidence extraction
3. **Deletion**: MUST be deleted immediately after digest completes (success or failure)
4. **Serialization guard**: `_raw_content` is prefixed with `_`, filtered by `public_metadata()`
5. **Logging guard**: Audit events NEVER include `_raw_content` value (only `_raw_content_size`)

### Test Assertions

- After digest phase completes, `source.metadata.get("_raw_content")` MUST be `None`
- `ResearchSource.to_dict()` MUST NOT include `_raw_content`
- `state.to_json()` MUST NOT include any `_raw_content` values

---

## Cache Key Specification

### Key Components (All Required)

| Component | Source | Purpose |
|-----------|--------|---------|
| `source_id` | `source.id` | Identity |
| `content_hash` | SHA256 of raw content, first 16 chars | Content change detection |
| `query_hash` | SHA256 of query string, first 8 chars | Query conditioning |
| `config_hash` | SHA256 of digest config tuple, first 8 chars | Config change detection |
| `impl_version` | `"1.0"` (bump on algorithm changes) | Algorithm versioning |

### Config Hash Inputs

```python
config_tuple = (
    config.deep_research_digest_policy,
    config.deep_research_digest_min_chars,
    config.deep_research_digest_max_sources,
    config.deep_research_digest_include_evidence,
    config.deep_research_digest_evidence_max_chars,
    config.deep_research_digest_max_evidence_snippets,
)
```

### Cache Key Format

```
digest:{impl_version}:{source_id}:{content_hash[:16]}:{query_hash[:8]}:{config_hash[:8]}
```

Example: `digest:1.0:src-abc12345:a1b2c3d4e5f6g7h8:q1w2e3r4:c5v6b7n8`

### Invalidation Rules

- Content changes → new `content_hash` → cache miss
- Query changes → new `query_hash` → cache miss
- Config changes → new `config_hash` → cache miss
- Algorithm changes → new `impl_version` → cache miss

### Circuit Breaker + Cache Interaction

- **Breaker open**: Cache READS allowed (serve existing digests), new digests SKIPPED
- **Timeout mid-digest**: Discard partial result, preserve original content, do NOT cache

---

## Canonical Text Pipeline

### Purpose

Define a single, deterministic text normalization that:
1. Produces consistent hashes across runs
2. Provides stable offsets for evidence locators
3. Is the exact format stored in archives

### Text Normalization Rules

**For HTML/Text content:**
```
1. Decode HTML entities (&#x27; → ', &amp; → &, etc.)
2. Strip all HTML tags (preserve inner text)
3. Normalize Unicode to NFC form
4. Collapse consecutive whitespace (spaces, tabs, newlines) to single space
5. Strip leading/trailing whitespace from result
6. Encode as UTF-8
```

**For PDF content:**
```
1. Extract text page-by-page using pypdf
2. Join pages with page separator: "\n\n---PAGE {n}---\n\n"
3. Normalize Unicode to NFC form
4. Collapse consecutive whitespace within each page to single space
5. Preserve page separators exactly (for locator calculation)
6. Strip leading/trailing whitespace from full result
7. Encode as UTF-8
```

### Canonical Text Properties

- **Idempotent**: Normalizing already-normalized text produces identical output
- **Hash-stable**: SHA256 of canonical text is reproducible across runs
- **Locator-compatible**: Character offsets reference positions in canonical text

### Archive Format

When archival is enabled:
- Store the **canonical text** (post-normalization)
- `source_text_hash` = SHA256 of canonical text bytes
- Evidence locator offsets reference canonical text positions

---

## Evidence Chunking Algorithm

### Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target chunk size | 400 characters | Fits typical evidence snippet cap |
| Max chunk size | 500 characters | Hard limit, truncate if exceeded |
| Min chunk size | 50 characters | Merge small chunks with previous |
| Overlap | 0 characters | No overlap (simpler offset tracking) |

### Boundary Rules

**Priority order for chunk boundaries:**
1. **Paragraph break** (`\n\n` or page separator) - highest priority
2. **Sentence end** (`. `, `! `, `? ` followed by capital or newline)
3. **Clause break** (`, `, `; `, `: `)
4. **Word break** (space)
5. **Hard cut** at max size (last resort, add `...` marker)

### Algorithm (HTML/Text)

```python
def chunk_text(canonical_text: str) -> list[tuple[str, int, int]]:
    """Returns list of (chunk_text, start_offset, end_offset)."""
    chunks = []
    start = 0
    while start < len(canonical_text):
        # Find chunk end within target..max range
        end = find_best_boundary(canonical_text, start + TARGET, start + MAX)
        chunk = canonical_text[start:end]
        if len(chunk) >= MIN or start == 0:
            chunks.append((chunk, start, end))
        else:
            # Merge with previous chunk
            prev_text, prev_start, _ = chunks.pop()
            chunks.append((prev_text + chunk, prev_start, end))
        start = end
    return chunks
```

### Algorithm (PDF)

Same as HTML/Text, but:
- Treat page separators (`---PAGE {n}---`) as paragraph breaks
- Track which page each chunk belongs to for locator generation
- Locator format: `page:{n}:char:{start_within_page}-{end_within_page}`

### Determinism

Given identical canonical text, chunking produces identical:
- Number of chunks
- Chunk boundaries (start/end offsets)
- Chunk content

---

## PDF Extraction Decision Table

### Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `deep_research_digest_fetch_pdfs` | `false` | Whether to fetch and extract PDF content |
| `deep_research_follow_links` | (existing) | Whether to follow links for content |

### Decision Matrix

| Source State | fetch_pdfs | Action | Ranking Input | Digest Eligible |
|--------------|------------|--------|---------------|-----------------|
| Has `content` (pre-extracted text) | any | Use existing | `content` | Yes |
| URL ends `.pdf`, no `content` | `true` | Fetch + Extract | Extracted text | Yes |
| URL ends `.pdf`, no `content` | `false` | Skip extraction | `snippet` only | No (no text) |
| `metadata.pdf_url` present, no `content` | `true` | Fetch + Extract | Extracted text | Yes |
| `metadata.pdf_url` present, no `content` | `false` | Skip extraction | `snippet` only | No (no text) |
| Non-PDF URL, no `content`, `follow_links=true` | any | Use Tavily Extract | Extracted text | Yes |
| Non-PDF URL, no `content`, `follow_links=false` | any | Skip extraction | `snippet` only | No (no text) |

### PDF Extraction Timing

**Key Decision**: PDF extraction happens in **ANALYSIS phase**, NOT in GATHER.

```
1. GATHER phase:
   - Sources collected with URL + snippet
   - NO content extraction yet (PDFs remain as URLs)

2. ANALYSIS phase start (BEFORE ranking):
   a. Content extraction pass:
      - For each source WITHOUT content:
        - If PDF URL and fetch_pdfs=true: Fetch + Extract PDF → set source.content
        - If non-PDF URL and follow_links=true: Tavily Extract → set source.content
        - Otherwise: Leave source.content empty

   b. Ranking pass (AFTER extraction):
      - Compute ranking features on source.content if available
      - If no content: Rank on source.snippet only, mark INELIGIBLE for digest

   c. Selection pass:
      - Sort by: quality DESC → relevance DESC → len(content) DESC → source_id ASC
      - Select top N sources that are ELIGIBLE for digest

   d. Digest pass:
      - Run DocumentDigestor.digest() on selected sources
      - Record fidelity, cleanup _raw_content

3. Budget allocation operates on digested content
```

**Why ANALYSIS not GATHER?**
- Keeps GATHER phase fast (no blocking on PDF downloads)
- Allows ranking to see extracted content before selection
- Extraction only happens for sources that pass quality filters

### Resource Limits for PDF Extraction

| Limit | Value | Behavior on Exceed |
|-------|-------|-------------------|
| Download size | 10 MB | Abort, emit warning, skip source |
| Page count | 500 pages | Extract first 500, emit warning |
| Extracted text | 500,000 chars | Truncate, emit warning |
| Memory per PDF | ~50 MB working | Use page-by-page extraction |
| Extraction timeout | 30 seconds | Abort, emit warning, skip source |

---

## Evidence Scoring Algorithm

### Specification

**Tokenization**: Split on whitespace and punctuation, lowercase, remove stopwords (English NLTK list)

**Scoring Formula**:
```
score = (matched_query_terms / total_query_terms) * term_weight_factor

term_weight_factor = sum(1/log2(doc_freq + 2) for matched_term) / matched_count
```

Where `doc_freq` is the count of chunks containing each term.

**Tie-breakers** (applied in order):
1. Higher score wins
2. Earlier chunk position wins (prefer content near document start)
3. Shorter chunk wins (prefer concise evidence)

**Empty/Short Query Fallback**:
- If query has <2 non-stopword terms: Use positional scoring only (first chunks win)
- Extract from document intro, key headings, and conclusion sections

### Determinism Guarantee

Given identical inputs (content, query, config), evidence extraction produces identical output (same snippets in same order with same scores).

---

## Component Ownership

### Responsibility Matrix

| Responsibility | Owner | Notes |
|----------------|-------|-------|
| Source ranking (quality, relevance) | Pipeline (`_execute_analysis_async`) | Before digest |
| Source selection (top N eligible) | Pipeline (`_execute_analysis_async`) | Before digest |
| Eligibility validation | DocumentDigestor | Safety check, may reject |
| Content extraction (PDF) | PDFExtractor | Called by digestor |
| Summarization | ContentSummarizer | Called by digestor |
| Evidence extraction | DocumentDigestor | Internal |
| Fidelity recording | Pipeline | After digest returns |

### Sequencing Contract

```
1. GATHER phase completes → sources have raw content
2. ANALYSIS phase starts:
   a. Compute ranking features on raw content (quality, relevance scores)
   b. Select top N eligible sources for digest (deterministic sort)
   c. For each selected source:
      - DocumentDigestor.digest() transforms content
      - Pipeline records fidelity
      - Pipeline cleans up _raw_content
   d. Budget allocation operates on digested content
3. ANALYSIS phase continues with compressed sources
```

### Concurrency Limits

- **Semaphore**: `deep_research_digest_max_concurrent` (default: 3)
- **Per-source timeout**: `deep_research_digest_timeout / max_concurrent`
- **Batch timeout**: `deep_research_digest_timeout` (overall)
- **Cancellation**: Propagate `asyncio.CancelledError`, cleanup `_raw_content`

---

## Phases

### Phase 1: Foundation - Models and Configuration

**Purpose**: Establish data structures, schemas, and configuration without changing existing behavior.

**Tasks**:
1. Define `DigestPayload` dataclass matching JSON schema above in `models.py`
2. Define `EvidenceSnippet` dataclass with `text`, `locator`, `relevance_score`
3. Add `FidelityLevel.DIGEST` to enum (after KEY_POINTS, before TRUNCATED)
4. Add `content_type: str = "text/plain"` field to `ResearchSource`
5. Add `is_digest` property: `return self.content_type == "digest/v1"`
6. Add digest configuration fields to `ResearchConfig`:
   - `deep_research_digest_policy`: off|auto|always (default: auto)
   - `deep_research_digest_min_chars`: int (default: 10000)
   - `deep_research_digest_max_sources`: int (default: 8)
   - `deep_research_digest_timeout`: float (default: 60.0)
   - `deep_research_digest_max_concurrent`: int (default: 3)
   - `deep_research_digest_include_evidence`: bool (default: true)
   - `deep_research_digest_evidence_max_chars`: int (default: 400)
   - `deep_research_digest_max_evidence_snippets`: int (default: 5)
   - `deep_research_digest_fetch_pdfs`: bool (default: false)
7. Add validation method `_validate_digest_config()` in ResearchConfig
8. Update `ResearchSource.to_dict()` to filter `_raw_content` and `_digest_*` keys

**Verification**: Unit tests for schema validation, config defaults, `is_digest` property, `to_dict()` filtering

**Fidelity Review**: Verify DigestPayload matches JSON schema, EvidenceSnippet has all fields, config validation rejects invalid values, `_raw_content` filtered from serialization

### Phase 2: PDF Extraction

**Purpose**: Enable text extraction from PDF sources with security hardening.

**Tasks**:
1. Add `pypdf` to pyproject.toml dependencies
2. Create `src/foundry_mcp/core/research/pdf_extractor.py`:
   - `PDFExtractor` class with `async extract(url_or_bytes) -> PDFExtractionResult`
   - `PDFExtractionResult`: `text`, `page_offsets: list[int]`, `warnings: list[str]`
   - Route URL fetching through existing hardened fetch path (allowlists, SSRF protections)
   - Validate content-type and magic bytes (`%PDF-`) before parsing
   - Size cap enforcement (max 10MB download, max 500 pages)
   - Chunked read with memory caps (load pages incrementally, not all at once)
   - Track page boundaries for `page:{n}:char:{start}-{end}` locators
   - Error handling with graceful fallback (return empty string + warning)
3. Add optional `pdfminer.six` fallback for complex PDFs (lazy import)
4. Add metrics: `pdf_extraction_duration_seconds`, `pdf_extraction_pages_total`
5. Add benchmark test for memory usage with large PDFs

**Verification**: Unit tests for PDF extraction (simple, complex, oversized, corrupted, SSRF blocked, magic byte validation)

**Fidelity Review**: Verify SSRF protections, size caps, page offset tracking, memory bounds, fallback behavior

### Phase 3: DocumentDigestor Core

**Purpose**: Implement the main digest logic with deterministic behavior.

**Tasks**:
1. Create `src/foundry_mcp/core/research/document_digest.py`:
   - `DigestResult` dataclass: `payload: DigestPayload`, `cache_hit: bool`, `duration_ms: int`
   - `DocumentDigestor` class with:
     - `__init__(summarizer: ContentSummarizer, pdf_extractor: PDFExtractor, config: ResearchConfig)`
     - `async digest(source: ResearchSource, query: str) -> DigestResult`
     - `is_eligible(source: ResearchSource) -> bool` - eligibility check
     - `_extract_evidence(chunks: list[str], chunk_offsets: list[int], query: str) -> list[EvidenceSnippet]`
     - `_compute_source_text_hash(content: str) -> str` - SHA256 hash
     - `serialize_payload(payload: DigestPayload) -> str` - JSON with schema validation
     - `deserialize_payload(content: str) -> DigestPayload | None` - safe deserialization
2. Implement eligibility logic per policy:
   - `off`: Always return False
   - `always`: Return True for any source with content
   - `auto`: Size >= min_chars AND quality in (HIGH, MEDIUM) AND not already digest
3. Implement evidence scoring per algorithm spec above
4. Reuse `ContentSummarizer.summarize()` with `SummarizationLevel.KEY_POINTS`
5. Generate locators: `char:{start}-{end}` for text, `page:{n}:char:{start}-{end}` for PDFs
6. Compute `source_text_hash` before any transformation

**Verification**: Unit tests for eligibility, evidence scoring determinism, locator format, hash computation, serialization round-trip

**Fidelity Review**: Verify scoring matches algorithm spec, locators are correct format, hash is SHA256, serialization validates schema

### Phase 4: Digest Caching

**Purpose**: Implement cache keys and invalidation per specification above.

**Tasks**:
1. Implement cache key generation per format: `digest:v1:{source_id}:{content_hash}:{query_hash}:{config_hash}`
2. Integrate with existing `SummaryCache`:
   - `_get_cached_digest(source, query, config) -> DigestPayload | None`
   - `_cache_digest(source, query, config, payload) -> None`
3. Add `_digest_cache_hit` metadata flag for observability
4. Add `DIGEST_IMPL_VERSION = "1.0"` constant (bump on algorithm changes)

**Verification**: Unit tests for cache key generation, cache hit/miss, invalidation on content/query/config change

**Fidelity Review**: Verify cache keys include all components, invalidation works correctly, impl_version is included

### Phase 5: Pipeline Integration

**Purpose**: Wire digest step into the deep research workflow per sequencing contract.

**Tasks**:
1. In `_execute_analysis_async()`, after source gathering:
   a. Compute ranking features (quality, relevance) on `source.content`
   b. Sort sources by: quality DESC → relevance DESC → len(content) DESC → source_id ASC
   c. Select top `max_sources` eligible for digest
2. For each selected source (with semaphore for concurrency):
   a. Set `source.metadata["_raw_content"] = source.content`
   b. Call `digestor.digest(source, query)`
   c. On success: Set `source.content = serialize_payload(result.payload)`
   d. On success: Set `source.content_type = "digest/v1"`
   e. Record fidelity with `FidelityLevel.DIGEST`
   f. **ALWAYS** delete `source.metadata["_raw_content"]` (success or failure)
3. Update budget allocation to use `source.content` length (now digest size)
4. Update citation formatting to use `EvidenceSnippet.text` when `source.is_digest`
5. Skip digest for sources where `source.is_digest == True` (subsequent iterations)

**Verification**: Integration tests for ranking→selection→digest order, `_raw_content` cleanup, citation with digest

**Fidelity Review**: Verify sequencing correct, `_raw_content` always deleted, citations use evidence snippets

### Phase 6: Observability

**Purpose**: Add audit events and metrics for monitoring.

**Tasks**:
1. Add audit events in DocumentDigestor:
   - `digest.started`: source_id, content_size (NOT content), policy, query_hash
   - `digest.completed`: source_id, original_chars, digest_chars, compression_ratio, cache_hit, duration_ms
   - `digest.skipped`: source_id, reason (not_eligible, already_digested, policy_off, breaker_open)
   - `digest.error`: source_id, error_type, error_message (sanitized)
   - `digest.circuit_breaker_triggered`: window_failures, window_size
2. Add metrics:
   - `digest_duration_seconds` (histogram, labels: policy, cache_hit)
   - `digest_sources_processed` (counter, labels: outcome=completed|skipped|error)
   - `digest_compression_ratio` (histogram)
   - `digest_cache_hits` (counter)
   - `digest_evidence_snippets` (histogram, count per digest)
3. Ensure NO audit event contains `_raw_content` value

**Verification**: Unit tests for audit event payloads, verify no raw content in logs

**Fidelity Review**: Verify all events fire correctly, metrics have correct labels, no raw content leaked

### Phase 7: Error Handling and Resilience

**Purpose**: Define explicit failure policies with robust circuit breaker.

**Tasks**:
1. Define error handling policy:
   - PDF extraction failure: Skip digest, preserve original content, emit warning, record in fidelity
   - Summarization failure: Skip digest, preserve original content, emit warning, record in fidelity
   - Timeout: Cancel in-progress work, preserve original content, do NOT cache partial results
2. Record failures in fidelity:
   - Set `FidelityLevel.FULL` (unchanged)
   - Add warning to `PhaseContentFidelityRecord.warnings`: "digest_failed: {reason}"
3. Implement sliding-window circuit breaker:
   - Track last N=10 digest attempts (success/failure)
   - Disable new digests if >70% fail AND at least 5 samples
   - Re-enable after 60 seconds OR new iteration starts
   - When open: Cache reads allowed, new digests skipped with `breaker_open` reason
4. Implement timeout budgets:
   - Per-source: `digest_timeout / max_concurrent`
   - Overall batch: `digest_timeout`
   - Use `asyncio.timeout()` with cancellation propagation

**Verification**: Unit tests for each failure scenario, circuit breaker trigger/reset/cache-read-while-open

**Fidelity Review**: Verify breaker logic, timeout propagation, no partial results cached

### Phase 8: Testing & Documentation

**Purpose**: Comprehensive test coverage and documentation.

**Tasks**:
1. Unit tests:
   - DigestPayload JSON schema validation (valid and invalid payloads)
   - EvidenceSnippet locator formats (all content types)
   - Evidence scoring algorithm (determinism, empty query fallback)
   - Digest eligibility (all policy/size/quality combinations)
   - PDF extraction (success, failure, SSRF, magic bytes, page offsets)
   - Cache key generation and invalidation scenarios
   - `_raw_content` lifecycle (creation, deletion, not serialized)
   - Circuit breaker (trigger threshold, auto-reset, cache-reads-while-open)
   - `source_text_hash` computation and archival linkage
2. Integration tests:
   - Full deep research with digest (mock summarizer + PDF extractor)
   - Ranking computed on raw content, digest applied after selection
   - Budget allocation uses digest size
   - Citations generated from evidence snippets with locators
   - Multi-iteration: digested sources not re-digested
   - Consumer branching on `content_type`
3. Contract tests:
   - Response envelope includes `content_fidelity` with DIGEST level
   - DigestPayload validates against JSON schema
   - `source_text_hash` matches archived content hash
4. Update fixtures and helpers:
   - Add DigestPayload fixtures
   - Update response envelope helpers for `content_type`
5. Update documentation:
   - `dev_docs/guides/deep-research.md`: Digest phase, evidence locators, caching
   - `dev_docs/configuration.md`: New config fields with defaults
   - `dev_docs/codebase_standards/mcp_response_schema.md`: DigestPayload contract

**Verification**: All tests pass, 100% coverage on new code, docs accurate

**Fidelity Review**: Verify all test categories covered, fixtures valid, documentation matches implementation

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Downstream consumers treat digest JSON as plain text | High | `content_type` field + `is_digest` property + contract tests |
| Citations break after digest | High | EvidenceSnippet contract with locators + `source_text_hash` traceability |
| `_raw_content` leaks into serialization/logs | High | Explicit lifecycle + `_` prefix filtering + test assertions |
| Stale digests served for different queries | High | `query_hash` in cache key + query-conditioned design |
| PDF extraction quality varies | Medium | pypdf + optional pdfminer.six fallback |
| Evidence snippets too large | Medium | Character caps (400) + snippet count limit (5) |
| Memory pressure from large PDFs | Medium | Size caps + chunked page loading |
| Circuit breaker too aggressive | Medium | Sliding window + minimum samples + cache-reads-while-open |
| Digest before ranking loses signals | High | Explicit sequencing: rank on raw → select → digest |

## Assumptions

1. ContentSummarizer's KEY_POINTS level produces suitable digest summaries
2. Existing archive infrastructure can store raw content by hash
3. pypdf license (BSD) is compatible with project
4. Simple keyword overlap scoring is sufficient for evidence relevance (no embeddings)
5. Existing tokenizer utility provides consistent counts

## Success Criteria

- [ ] DigestPayload validates against JSON schema v1.0
- [ ] Evidence locators follow per-content-type format
- [ ] `source_text_hash` enables citation verification against archives
- [ ] Large sources (>10k chars) digested with >50% compression
- [ ] Consumer code branches on `content_type` / `is_digest`
- [ ] Citations use evidence snippets with locators
- [ ] `_raw_content` never appears in serialization or logs
- [ ] Cache keys include query_hash (query-conditioned)
- [ ] Fidelity records show DIGEST level with accurate tokens
- [ ] Budget allocation uses compressed size
- [ ] Ranking features computed on raw text before digest
- [ ] Circuit breaker allows cache reads when open
- [ ] All tests pass with 100% coverage on new code
