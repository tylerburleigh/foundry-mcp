# AI Consultation Layer Plan (Revised)

> Restore LLM-backed capabilities in foundry-mcp without `claude_skills` dependency.

## 1. Existing Infrastructure (Build On This)

### Provider Registry (`src/foundry_mcp/core/providers/registry.py`)
- `register_provider()`, `resolve_provider()`, `available_providers()`
- Lazy loading, dependency injection, availability checks
- Already has: gemini, codex, cursor-agent, claude, opencode providers

### Provider Base (`src/foundry_mcp/core/providers/base.py`)
- `ProviderContext` ABC with `generate(request) -> ProviderResult`
- `ProviderRequest` (prompt, system_prompt, model, timeout, etc.)
- `ProviderResult` (content, status, tokens, duration_ms)
- `ProviderHooks` for lifecycle events (before_execute, on_stream_chunk, after_result)
- Error hierarchy: `ProviderError`, `ProviderUnavailableError`, `ProviderTimeoutError`

### Detectors (`src/foundry_mcp/core/providers/detectors.py`)
- `detect_provider_availability()` for CLI probes
- Environment overrides (`FOUNDRY_{PROVIDER}_AVAILABLE_OVERRIDE`)
- Test mode support (`FOUNDRY_PROVIDER_TEST_MODE`)

---

## 2. Documentation Strategy

| Artifact | Source | AI Required? |
|----------|--------|--------------|
| `codebase.json` | AST parsing | **No** - deterministic, reproducible |
| `project-overview.md` | AI narrative | **Yes** - requires provider, no fallback |
| `architecture.md` | AI narrative | **Yes** - requires provider, no fallback |
| `component-inventory.md` | AI narrative | **Yes** - requires provider, no fallback |
| `doc-generation-state.json` | Generator state | **No** - metadata only |

**Note:** If no AI provider is available, doc generation fails with `AI_NO_PROVIDER` error and remediation guidance ("Install gemini/codex/cursor-agent CLI").

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI Commands / MCP Tools                                       │
│  (review.py, docgen.py, tools/review_tools.py)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  AI Consultation Layer                                          │
│  src/foundry_mcp/core/ai_consultation.py                       │
│  ├── ConsultationOrchestrator                                  │
│  │   ├── run_doc_generation_consultation()                     │
│  │   ├── run_plan_review_consultation()                        │
│  │   └── run_fidelity_review_consultation()                    │
│  ├── PromptRegistry (versioned prompts)                        │
│  ├── ResultCache (persistent, scoped)                          │
│  └── ConsultationResult (typed response)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Provider Registry (EXISTING)                                   │
│  src/foundry_mcp/core/providers/                               │
│  ├── registry.py (resolve_provider, available_providers)       │
│  ├── base.py (ProviderContext, ProviderRequest, ProviderResult)│
│  └── detectors.py (availability probes)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Provider Implementations (EXISTING)                            │
│  gemini.py, codex.py, cursor_agent.py, claude.py, opencode.py  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Module Design

### 4.1 `src/foundry_mcp/core/ai_consultation.py`

```python
@dataclass(frozen=True)
class ConsultationRequest:
    """Normalized request for AI consultation workflows."""
    workflow: str  # "doc_generation" | "plan_review" | "fidelity_review"
    prompt_id: str  # Reference to versioned prompt
    context: Dict[str, Any]  # Workflow-specific context
    provider_id: Optional[str] = None  # None = auto-select
    model: Optional[str] = None
    cache_key: Optional[str] = None
    timeout: Optional[float] = 300.0

@dataclass(frozen=True)
class ConsultationResult:
    """Normalized response from AI consultation."""
    workflow: str
    content: str
    provider_id: str
    model_used: str
    tokens: TokenUsage
    duration_ms: float
    cache_hit: bool
    raw_payload: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

class ConsultationOrchestrator:
    """Orchestrates AI consultations using the provider registry."""

    def __init__(
        self,
        prompt_registry: PromptRegistry,
        cache: Optional[ResultCache] = None,
        provider_priority: Optional[List[str]] = None,
    ): ...

    def consult(self, request: ConsultationRequest) -> ConsultationResult:
        """Execute a consultation with provider selection and caching."""
        ...

    def is_available(self) -> bool:
        """Check if any AI provider is available."""
        ...
```

### 4.2 Prompt Registry

```python
# src/foundry_mcp/core/prompts/__init__.py

@dataclass(frozen=True)
class PromptTemplate:
    """Versioned prompt template."""
    id: str  # e.g., "doc_gen.project_overview.v1"
    version: str  # Semantic version
    system_prompt: str
    user_template: str  # Supports {variable} substitution
    required_context: Set[str]  # Required context keys
    metadata: Dict[str, Any] = field(default_factory=dict)

# Prompts stored as Python modules:
# src/foundry_mcp/core/prompts/
# ├── __init__.py
# ├── doc_generation.py  # DOC_GEN_PROJECT_OVERVIEW_V1, etc.
# ├── plan_review.py     # PLAN_REVIEW_FULL_V1, PLAN_REVIEW_SECURITY_V1, etc.
# └── fidelity_review.py # FIDELITY_REVIEW_V1, etc.
```

### 4.3 Caching

```python
# Cache location: .cache/foundry-mcp/consultations/
# Structure: {workflow}/{spec_id_or_project_hash}/{cache_key}.json

@dataclass
class CacheEntry:
    """Cached consultation result."""
    result: ConsultationResult
    created_at: str  # ISO timestamp
    expires_at: Optional[str]  # ISO timestamp, None = never
    context_hash: str  # Hash of context for invalidation
    git_commit: Optional[str]  # Git commit hash when created
```

---

## 5. Error Codes

| Code | Type | Description | Remediation |
|------|------|-------------|-------------|
| `AI_NO_PROVIDER` | unavailable | No AI provider available | Install gemini/codex/cursor-agent CLI |
| `AI_PROVIDER_TIMEOUT` | unavailable | Provider execution timed out | Retry with --timeout or smaller scope |
| `AI_PROVIDER_ERROR` | internal | Provider returned error | Check provider logs, retry |
| `AI_CONTEXT_TOO_LARGE` | validation | Context exceeds provider limits | Reduce scope or use batching |
| `AI_PROMPT_NOT_FOUND` | not_found | Unknown prompt ID | Check workflow configuration |
| `AI_CACHE_STALE` | validation | Cached result outdated | Use --no-cache to regenerate |

---

## 6. Implementation Phases

### Phase A – Foundation
- [ ] Create `src/foundry_mcp/core/ai_consultation.py` with core classes
- [ ] Create `src/foundry_mcp/core/prompts/` with initial templates
- [ ] Implement `ConsultationOrchestrator` using existing provider registry
- [ ] Add `ResultCache` with filesystem persistence
- [ ] Add CLI flags: `--ai-provider`, `--no-cache`
- [ ] Define error codes in `src/foundry_mcp/core/responses.py`

### Phase B – Documentation Generation
- [ ] Port prompts from `claude_skills.llm_doc_gen` to `prompts/doc_generation.py`
- [ ] Extend `docgen.py` to call consultation layer for markdown artifacts:
  - Keep AST parsing deterministic → `codebase.json` unchanged
  - Replace `_build_project_overview()`, `_build_architecture_doc()`, `_build_component_inventory()` with AI-enhanced versions
  - Pass `codebase.json` content as context to AI prompts
  - Error with `AI_NO_PROVIDER` if no provider available (no fallback)
- [ ] Remove current placeholder markdown generators
- [ ] Integrate with CLI `llm-doc generate` command
- [ ] Add MCP tool `spec-doc-llm` integration
- [ ] Add tests with mocked providers

### Phase C – Plan Review
- [ ] Port prompts from `claude_skills.sdd_plan_review` to `prompts/plan_review.py`
- [ ] Update `review.py` to use consultation layer for full/security/feasibility types
- [ ] Update MCP `spec-review` tool
- [ ] Error with remediation when AI unavailable

### Phase D – Fidelity Review
- [ ] Port prompts from `claude_skills.sdd_fidelity_review` to `prompts/fidelity_review.py`
- [ ] Implement `review fidelity` CLI using consultation layer
- [ ] Update MCP `spec-review-fidelity` tool
- [ ] Support incremental/file-based scopes

### Phase E – Cleanup & Documentation
- [ ] Update specs (`specs/active/*.json`), capability manifests
- [ ] Add fixture tests with recorded provider responses
- [ ] Remove remaining `claude_skills` references
- [ ] Document configuration in `docs/guides/ai-consultation.md`

---

## 7. Configuration

```bash
# Provider selection (priority order)
export FOUNDRY_AI_PROVIDER_PRIORITY="gemini,codex,claude"

# Cache configuration
export FOUNDRY_AI_CACHE_DIR=".cache/foundry-mcp/consultations"
export FOUNDRY_AI_CACHE_TTL_HOURS=24

# Per-provider overrides (existing)
export FOUNDRY_GEMINI_AVAILABLE_OVERRIDE=true
```

---

## 8. Testing Strategy

1. **Unit tests**: Mock `ProviderContext.generate()` to return fixed responses
2. **Integration tests**: Use `FOUNDRY_PROVIDER_TEST_MODE=1` with response fixtures
3. **Fixture format**: Record actual provider responses, replay in tests
4. **Schema validation**: Ensure all responses comply with MCP envelope format

---

## 9. Files to Create/Modify

**Create:**
- `src/foundry_mcp/core/ai_consultation.py` - Orchestrator and core types
- `src/foundry_mcp/core/prompts/__init__.py` - PromptRegistry
- `src/foundry_mcp/core/prompts/doc_generation.py` - Doc gen prompts
- `src/foundry_mcp/core/prompts/plan_review.py` - Plan review prompts
- `src/foundry_mcp/core/prompts/fidelity_review.py` - Fidelity prompts
- `docs/guides/ai-consultation.md` - Configuration guide

**Modify:**
- `src/foundry_mcp/core/docgen.py` - Replace placeholder markdown with AI-enhanced generation
- `src/foundry_mcp/cli/commands/docgen.py` - Add `--ai-provider`, `--no-cache` flags
- `src/foundry_mcp/cli/commands/review.py` - Use consultation layer
- `src/foundry_mcp/tools/review_tools.py` - MCP tool updates
- `src/foundry_mcp/core/responses.py` - Add AI error codes

---

## 10. Key Design Decisions

| Decision | Resolution |
|----------|------------|
| Provider infrastructure | Use existing `providers/` registry, not `llm_provider.py` |
| Prompt storage | Python modules under `core/prompts/` with dataclass templates |
| Caching layout | `.cache/foundry-mcp/consultations/{workflow}/{scope}/{key}.json` |
| Streaming | Not for MCP tools (final payloads only); optional for CLI via hooks |
| Deterministic boundary | AST parsing (`codebase.json`) stays deterministic; markdown is AI-only |
| Fallback behavior | None - AI provider required; error with remediation if unavailable |
