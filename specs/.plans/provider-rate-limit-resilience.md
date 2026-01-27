# provider-rate-limit-resilience

## Mission

Add robust rate limit resilience to all search providers by implementing jitter, per-provider rate limiting, and circuit breakers while eliminating code duplication.

## Objective

Improve the resilience of all 5 search providers (Tavily, Google, Perplexity, Semantic Scholar, Tavily Extract) to rate limiting and transient failures by:
1. Adding jitter to exponential backoff to prevent thundering herd
2. Enforcing per-provider rate limits using existing `TokenBucketLimiter`
3. Adding per-provider circuit breakers using existing `CircuitBreaker`
4. Refactoring ~450 lines of duplicated `_execute_with_retry()` code into a shared utility

## Scope

### In Scope
- Create shared `providers/resilience.py` module composing existing primitives
- Refactor all 5 providers to use shared resilience utilities
- Add per-provider circuit breakers with configurable thresholds
- Add per-provider rate limiting using TokenBucketLimiter
- Enhance deep research orchestration with circuit breaker awareness
- Add observability/audit logging for resilience events
- Unit and integration tests for new resilience module
- Feature flag for safe rollout

### Out of Scope
- Changes to existing `resilience.py` or `rate_limit.py` core modules
- Modifying the SearchProvider base class interface (backward compatible)
- Adding new external dependencies
- Per-tenant rate limiting (global per-provider only)
- Distributed/cross-process rate limiting (per-process only)

---

## Behavior Contract

### Error Taxonomy and Classification

Each error type has defined handling semantics:

| Error Type | HTTP Status | Retryable | Trips Breaker | Backoff | Short-Circuit |
|------------|-------------|-----------|---------------|---------|---------------|
| Rate Limit | 429 | Yes | No | Use Retry-After or exponential | No |
| Server Error | 5xx | Yes | Yes | Exponential + jitter | No |
| Timeout | N/A | Yes | Yes | Exponential + jitter | No |
| Connection Error | N/A | Yes | Yes | Exponential + jitter | No |
| Auth Error | 401, 403 (non-quota) | No | No | N/A | Yes |
| Validation Error | 400 | No | No | N/A | Yes |
| Quota Error | 403 (Google quota) | Yes | Yes | Exponential + jitter | No |

**Provider Override Hook**: Each provider implements `classify_error(response, exception) -> ErrorClassification` to handle provider-specific semantics (e.g., Google's 403 quota detection).

```python
@dataclass
class ErrorClassification:
    retryable: bool
    trips_breaker: bool
    backoff_seconds: Optional[float] = None  # None = use default exponential
    error_type: str = "unknown"
```

### Rate Limiter Policy

- **Behavior**: Wait (block) until token available, not reject
- **Max Wait**: Configurable per-provider, default 30 seconds
- **Timeout Integration**: Limiter wait counts against request time budget
- **Ordering**: Acquire limiter token → make request → apply backoff only on failure

```python
@dataclass
class RateLimiterConfig:
    requests_per_second: float = 1.0
    burst_limit: int = 5
    max_wait_seconds: float = 30.0  # Fail if would wait longer
```

### Circuit Breaker Semantics

- **States**: CLOSED (normal) → OPEN (failing) → HALF_OPEN (probing)
- **Threshold**: 5 consecutive failures opens circuit (configurable)
- **Recovery**: 30 seconds in OPEN before transitioning to HALF_OPEN
- **Half-Open**: Allow 1 probe request; success → CLOSED, failure → OPEN
- **Deep Research Selection**: Skip OPEN providers; include HALF_OPEN (allows recovery probe)

**State Inspection API**:
```python
class ProviderResilienceManager:
    def get_breaker_state(self, provider: str) -> CircuitState  # CLOSED|OPEN|HALF_OPEN
    def is_provider_available(self, provider: str) -> bool  # True if CLOSED or HALF_OPEN
    def get_provider_status(self, provider: str) -> ProviderStatus  # Full status dict
```

### Time Budget and Cancellation

- **Per-Request Budget**: Default 60 seconds total (configurable)
- **Budget Accounting**: Limiter wait + retry attempts + backoff delays all count
- **Cancellation**: `asyncio.CancelledError` propagates immediately through all waits
- **Budget Exceeded**: Raises `TimeBudgetExceededError` with elapsed time and attempts made

---

## API Surface

### ProviderResilienceManager Lifecycle

- **Pattern**: Module-level singleton with lazy initialization
- **Scope**: Per-process (not distributed/cross-process)
- **Event Loop Safety**: Uses `asyncio.Lock` for thread-safe async access
- **Test Reset**: `ProviderResilienceManager.reset_for_testing()` clears all state

```python
# Module-level access
_manager: Optional[ProviderResilienceManager] = None

def get_resilience_manager() -> ProviderResilienceManager:
    global _manager
    if _manager is None:
        _manager = ProviderResilienceManager()
    return _manager

def reset_resilience_manager_for_testing() -> None:
    global _manager
    _manager = None
```

### Configuration Precedence

1. Explicit parameter in `execute_with_resilience()` call
2. Provider's `resilience_config` property
3. `PROVIDER_CONFIGS` defaults dict
4. Global defaults

### Provider Override Hook

```python
class SearchProvider(ABC):
    def classify_error(
        self,
        response: Optional[httpx.Response],
        exception: Optional[Exception]
    ) -> ErrorClassification:
        """Override to customize error handling for this provider."""
        # Default implementation uses error taxonomy table
        ...
```

---

## Observability Event Schema

All resilience events use consistent structure for queryable telemetry:

```python
@dataclass
class ResilienceEvent:
    event_type: str  # "rate_limit_wait", "retry_attempt", "circuit_state_change"
    provider: str
    timestamp: datetime

    # Optional fields by event type
    attempt: Optional[int] = None
    max_attempts: Optional[int] = None
    wait_ms: Optional[int] = None
    breaker_state: Optional[str] = None
    error_type: Optional[str] = None
    error_classification: Optional[str] = None
    budget_remaining_ms: Optional[int] = None
    correlation_id: Optional[str] = None
```

**Event Types**:
- `rate_limit_wait` - Waiting for rate limiter token
- `retry_attempt` - Starting retry after failure
- `circuit_state_change` - Breaker state transition
- `budget_exceeded` - Time budget exhausted
- `request_success` - Successful request (for metrics)
- `request_failure` - Final failure after retries

---

## Rollout and Compatibility

### Feature Flag

```python
# Environment variable controls new resilience path
FOUNDRY_ENHANCED_RESILIENCE = os.environ.get("FOUNDRY_ENHANCED_RESILIENCE", "true").lower() == "true"
```

- **Default**: Enabled (`true`) for new installs
- **Fallback**: Set to `false` to use legacy per-provider retry logic
- **Per-Provider Override**: `FOUNDRY_ENHANCED_RESILIENCE_TAVILY=false` etc.

### Backward Compatibility

- No changes to `SearchProvider.search()` interface
- Providers continue to raise same exception types (`RateLimitError`, `AuthenticationError`, etc.)
- Default configs match current behavior (3 retries, ~1 req/s effective rate)

### Acceptance Criteria (Latency)

- P50 latency increase < 5% under normal conditions
- P99 latency acceptable increase under rate limiting (due to proper backoff)

---

## Phases

### Phase 1: Create Shared Provider Resilience Module

**Purpose**: Create centralized resilience utilities that compose existing primitives for search providers

**Tasks**:
1. Create `src/foundry_mcp/core/research/providers/resilience.py` with:
   - `ProviderResilienceConfig` dataclass with per-provider defaults
   - `ErrorClassification` dataclass for error taxonomy
   - `ProviderResilienceManager` singleton with lifecycle management
   - `async_retry_with_backoff()` - async version with injectable RNG/clock for testing
   - `execute_with_resilience()` - unified executor with time budget support
2. Add `PROVIDER_CONFIGS` dict with tuned defaults for each provider
3. Add state inspection API (`get_breaker_state()`, `is_provider_available()`)
4. Integrate with `audit_log()` using defined event schema
5. Add `reset_for_testing()` mechanism
6. Create unit tests for all new utilities in `tests/core/research/providers/test_resilience.py`

**Verification**:
- Unit tests pass for all new utilities
- Jitter tests are deterministic (seeded RNG)
- Manager creates isolated limiters/breakers per provider
- Test reset clears all state correctly

**Fidelity Review**: Run `foundry-review` to verify implementation matches spec for:
- All dataclasses match defined schemas
- Manager lifecycle matches singleton pattern
- Error taxonomy table implemented correctly
- Observability events match schema

### Phase 2: Refactor Tavily Provider (Pilot)

**Purpose**: Refactor Tavily as pilot to validate the shared resilience approach before rolling out to other providers

**Tasks**:
1. Add feature flag check to enable/disable enhanced resilience
2. Implement `classify_error()` override for Tavily-specific semantics
3. Replace `_execute_with_retry()` in `tavily.py` with call to `execute_with_resilience()`
4. Add `resilience_config` property returning Tavily-specific config
5. Preserve existing error classification (401 → AuthenticationError, 429 → RateLimitError)
6. Update existing Tavily tests to work with new implementation

**Verification**:
- All existing Tavily tests pass with flag enabled AND disabled
- Manual test with rate limiting shows jittered backoff in logs
- Circuit breaker opens after configured failure threshold
- Time budget enforced correctly

**Fidelity Review**: Run `foundry-review` to verify:
- Feature flag integration correct
- Error classification matches taxonomy table
- Existing test coverage maintained
- No behavioral regressions vs legacy path

### Phase 3: Refactor Remaining Providers

**Purpose**: Apply the validated pattern to all remaining search providers

**Tasks**:
1. Refactor `google.py` with `classify_error()` handling 403 quota detection
2. Refactor `perplexity.py` `_execute_with_retry()`
3. Refactor `semantic_scholar.py` `_execute_with_retry()`
4. Refactor `tavily_extract.py` `_execute_with_retry()`
5. Add `resilience_config` property to each provider
6. Ensure each provider's `classify_error()` preserves provider-specific semantics

**Verification**:
- All provider unit tests pass
- Each provider has isolated rate limiter and circuit breaker
- Total lines reduced by ~350-400 (duplicate code eliminated)
- Google 403 quota errors correctly classified as retryable

**Fidelity Review**: Run `foundry-review` to verify:
- All 5 providers use shared `execute_with_resilience()`
- Provider-specific `classify_error()` implementations match documented behavior
- Code duplication eliminated as expected
- No provider-specific edge cases missed

### Phase 4: Enhance Deep Research Orchestration

**Purpose**: Make deep research gathering phase aware of circuit breaker states for smarter provider selection

**Tasks**:
1. Update `deep_research.py` gathering phase to:
   - Use `is_provider_available()` to filter providers (skips OPEN, allows HALF_OPEN)
   - Log audit event when all providers unavailable
   - Include circuit breaker states in gathering result metadata
2. Add graceful degradation when circuit breaker opens mid-gathering
3. Respect half-open semantics (allow recovery probes)

**Verification**:
- Deep research skips providers with OPEN circuit breakers
- Deep research allows HALF_OPEN providers (recovery probes)
- Audit log shows `all_providers_circuit_open` when applicable
- Gathering continues successfully when some providers fail

**Fidelity Review**: Run `foundry-review` to verify:
- `is_provider_available()` used correctly for provider filtering
- HALF_OPEN semantics respected (recovery probes allowed)
- Audit events match observability schema
- Graceful degradation behavior matches spec

### Phase 5: Testing, Documentation, and Rollout

**Purpose**: Ensure comprehensive test coverage, documentation, and safe rollout

**Tasks**:
1. Add comprehensive integration tests:
   - Deterministic jitter tests (seeded RNG)
   - Circuit breaker threshold tests
   - Rate limiter enforcement tests
   - Time budget tests
   - Deep research provider failover tests
2. Update provider docstrings with resilience behavior notes
3. Add CHANGELOG entry documenting new resilience features
4. Update any affected fixture files
5. Document feature flag usage for operators
6. Verify feature flag rollback works correctly

**Verification**:
- All new tests pass
- Code coverage for resilience module >90%
- Tests are deterministic (no flaky CI)
- Manual end-to-end test with Semantic Scholar (no API key) shows resilience working
- Feature flag rollback tested and working

**Fidelity Review**: Run `foundry-review` for full spec to verify:
- All success criteria met
- All phases implemented as specified
- Documentation complete
- No spec deviations

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing provider behavior | High | Feature flag for rollback, pilot with Tavily first |
| Circuit breaker too aggressive | Medium | Conservative defaults (5 failures, 30s recovery), configurable |
| Jitter causing longer delays | Low | Jitter is 50-150% of base delay, still bounded by max_delay |
| Thread safety in manager | Medium | Use asyncio.Lock for all shared state access |
| Flaky tests from randomness | Medium | Injectable seeded RNG and fake clock |
| Time budget too restrictive | Medium | Generous default (60s), configurable per-request |

## Assumptions

- Existing `resilience.py` CircuitBreaker and `rate_limit.py` TokenBucketLimiter are stable and tested
- Provider rate limits (1 req/s default) are appropriate starting points
- Deep research can tolerate slightly slower provider responses due to rate limiting
- Per-process rate limiting is sufficient (no distributed coordination needed)

## Success Criteria

- [ ] All 5 providers use shared `execute_with_resilience()` utility
- [ ] Jitter is applied to all retry delays (verified by deterministic test)
- [ ] Per-provider rate limiting enforced via TokenBucketLimiter
- [ ] Per-provider circuit breakers prevent hammering failing providers
- [ ] Error taxonomy correctly classifies all error types per provider
- [ ] Time budget prevents unbounded request duration
- [ ] Deep research gracefully handles circuit breaker open states
- [ ] ~400 lines of duplicate code eliminated
- [ ] All existing tests pass + new resilience tests added
- [ ] Audit logs capture events with consistent schema
- [ ] Feature flag allows rollback to legacy behavior
- [ ] CHANGELOG and documentation updated
- [ ] Fidelity reviews pass for all phases

## Critical Files

| File | Role |
|------|------|
| `src/foundry_mcp/core/research/providers/resilience.py` | NEW - shared utilities |
| `src/foundry_mcp/core/research/providers/tavily.py` | Pilot refactor |
| `src/foundry_mcp/core/research/providers/google.py` | Refactor |
| `src/foundry_mcp/core/research/providers/perplexity.py` | Refactor |
| `src/foundry_mcp/core/research/providers/semantic_scholar.py` | Refactor |
| `src/foundry_mcp/core/research/providers/tavily_extract.py` | Refactor |
| `src/foundry_mcp/core/research/providers/base.py` | Add classify_error() hook |
| `src/foundry_mcp/core/research/workflows/deep_research.py` | Orchestration updates |
| `src/foundry_mcp/core/resilience.py` | Existing - compose from |
| `src/foundry_mcp/core/rate_limit.py` | Existing - compose from |
| `tests/core/research/providers/test_resilience.py` | NEW - tests |
| `CHANGELOG.md` | Document new features |
