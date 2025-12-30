# Test Coverage Parity: AI Consultation & Research Router

## Objective

Bring test coverage to parity between AI Consultation and Research Router modules, then add workflow/orchestrator tests and E2E tests with mocked providers for both.

## Current State (After This Session)

### Research Router Tests ✅
- `tests/unit/test_core/research/test_models.py` - 53 tests (Pydantic models, enums)
- `tests/unit/test_core/research/test_memory.py` - 49 tests (storage CRUD, TTL, concurrency)
- `tests/unit/test_core/research/test_workflows.py` - 32 tests (NEW - workflow classes with mocked providers)
- `tests/tools/unified/test_research.py` - 47 tests (router dispatch, mocked workflows)

### AI Consultation Tests
- `tests/unit/test_ai_consultation.py` - Tests for:
  - ✅ ConsultationWorkflow enum
  - ✅ ConsultationRequest dataclass
  - ✅ ConsultationResult dataclass
  - ✅ ProviderResponse dataclass
  - ✅ AgreementMetadata dataclass
  - ✅ ConsensusResult dataclass
  - ❌ ResolvedProvider dataclass (NOT tested)
  - ❌ ResultCache class (NOT tested)
  - ❌ ConsultationOrchestrator class (NOT tested)

## Remaining Work

### 1. Add Missing AI Consultation Dataclass Tests
File: `tests/unit/test_ai_consultation.py`

Add tests for:
- `ResolvedProvider` dataclass (creation, attributes)
- `ResultCache` class (init, cache operations, TTL, file I/O)

### 2. Add ConsultationOrchestrator Tests
File: `tests/unit/test_ai_consultation.py`

Add tests for:
- `ConsultationOrchestrator.is_available()`
- `ConsultationOrchestrator.consult()` with mocked providers
- `ConsultationOrchestrator._execute_single_provider()`
- `ConsultationOrchestrator._execute_multi_provider()`
- Error handling (timeout, provider unavailable)
- Response structure validation

### 3. Add E2E Tests with Mocked Providers

#### For AI Consultation
File: `tests/integration/test_ai_consultation_e2e.py`

Test full flow:
- Plan review workflow end-to-end
- Fidelity review workflow end-to-end
- Multi-model consensus with fallback
- Cache hit/miss scenarios

#### For Research Router
File: `tests/integration/test_research_e2e.py`

Test full flow through router:
- `research action=chat` -> ChatWorkflow -> response envelope
- `research action=consensus` -> ConsensusWorkflow -> response envelope
- `research action=thinkdeep` -> ThinkDeepWorkflow -> response envelope
- `research action=ideate` -> IdeateWorkflow -> response envelope
- Thread persistence across calls
- Feature flag gating

## Key Files

### AI Consultation
- Source: `src/foundry_mcp/core/ai_consultation.py`
- Tests: `tests/unit/test_ai_consultation.py`

### Research Router
- Source: `src/foundry_mcp/core/research/` (models, memory, workflows)
- Source: `src/foundry_mcp/tools/unified/research.py` (router)
- Tests: `tests/unit/test_core/research/` (models, memory, workflows)
- Tests: `tests/tools/unified/test_research.py` (router)

## Test Patterns to Follow

### Mocking Providers
```python
from foundry_mcp.core.providers import ProviderResult, ProviderStatus, TokenUsage

mock_context = MagicMock()
mock_context.generate.return_value = ProviderResult(
    content="Mock response",
    status=ProviderStatus.SUCCESS,
    provider_id="gemini",
    model_used="gemini-2.0-flash",
    tokens=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
    duration_ms=500.0,
)
```

### Response Structure Checks
```python
assert result.success is True
assert isinstance(result.content, str)
assert result.metadata["key"] == expected_value
assert result.meta["version"] == "response-v2"  # For router responses
```

## PR Info
- Branch: `sandbox/foundry-mcp-20251229-0843`
- PR: https://github.com/tylerburleigh/foundry-mcp/pull/9
