# Resilience Improvements

## Objective

Improve the resilience, observability, and operational behavior of background tasks and provider execution in foundry-mcp, with a focus on deep research workflows. Enable proper cancellation propagation, timeout enforcement, progress visibility, and executor isolation.

## Mission

Add robust cancellation, timeout, and progress reporting to background tasks and provider calls.

## Scope

### In Scope
- Cancellation propagation through deep research workflow phases and provider calls
- Timeout watchdog for background tasks with state persistence
- Default timeout configuration for deep research
- Progress heartbeats and interim state persistence
- Executor isolation for provider calls (if blocking operations identified)
- Provider-level timeout enforcement across all providers
- Tests and fixtures for new functionality
- Documentation updates

### Out of Scope
- Changes to synchronous/blocking API endpoints
- Provider retry logic (separate concern)
- Rate limiting (separate concern)
- Backward compatibility shims

## References
- dev_docs/mcp_best_practices/12-timeout-resilience.md
- dev_docs/mcp_best_practices/15-concurrency-patterns.md
- dev_docs/mcp_best_practices/05-observability-telemetry.md
- dev_docs/mcp_best_practices/07-error-semantics.md
- dev_docs/codebase_standards/mcp_response_schema.md

## Design Decisions

### Task Registry
- **Storage**: In-memory `Dict[str, BackgroundTask]` keyed by task_id, stored in a singleton `TaskRegistry` class.
- **Thread safety**: Use `asyncio.Lock` for all registry mutations (add, remove, lookup).
- **Cleanup**: Tasks removed from registry 5 minutes after reaching terminal state (completed, cancelled, timed_out, failed).
- **Lookup**: `TaskRegistry.get(task_id) -> Optional[BackgroundTask]` returns None if not found or expired.

### Cancellation Mechanism (Two-Phase)
- **Phase 1 - Cooperative (0-5s)**: Set `cancellation_event`, allow workflow to check at iteration boundaries and exit gracefully.
- **Phase 2 - Forced (after 5s)**: If task hasn't terminated, call `asyncio.Task.cancel()` to raise `CancelledError`.
- **CancelledError handling**: Catch at top-level workflow, write `cancelling` state, perform cleanup, then write `cancelled` state.
- **Resource cleanup**: Use `try/finally` blocks; cleanup runs regardless of cooperative vs forced cancellation.
- **Partial results**: Discard partial results from interrupted provider calls; persist only completed iterations.

### Atomic State Transitions
- **On cancellation request**: Immediately write `cancelling` state before any cleanup.
- **On cleanup complete**: Write `cancelled` state.
- **On crash recovery**: If task found in `cancelling` state on startup, transition to `cancelled` (cleanup assumed incomplete but safe).
- **On timeout**: Write `timing_out` -> perform cleanup -> write `timed_out`.

### Timeout Layering
- **Task-level timeout** (`task_timeout`): Overall deadline for the entire background task, enforced by active watchdog.
- **Provider-level timeout** (`request.timeout`): Per-call deadline for individual provider invocations.
- **Relationship**: Provider timeout should be < task timeout. Task timeout cancels all in-flight provider calls when triggered (uses forced cancellation).
- **Concurrent provider calls**: On cancellation/timeout, all concurrent calls are cancelled (no completion of in-flight calls).

### Timeout Watchdog (Active)
- **Mechanism**: Background asyncio task that polls all running tasks every 10 seconds.
- **On timeout detected**: Initiate two-phase cancellation with `timed_out` as terminal state.
- **Lifecycle**: Started on application startup, cancelled on shutdown.

### Heartbeat Staleness
- **Expected interval**: Heartbeat updated at least every 30 seconds during active work.
- **Update points**: Phase start, phase end, before each provider call, iteration boundaries.
- **Long calls**: Heartbeat updated BEFORE each provider call; if call takes >60s, staleness is expected.
- **Staleness threshold**: Task marked "stale" if no heartbeat for 90 seconds (3x interval) while status is `running`.
- **Staleness vs timeout**: Staleness is informational; timeout is terminal. Stale tasks continue running; timed-out tasks are cancelled.

### ProviderTimeoutError Handling
- **At workflow level**: Catch `ProviderTimeoutError` at iteration/sub-query level.
- **Behavior**: Log the failure, mark that sub-query as failed, continue with remaining work.
- **Final report**: Include list of failed sub-queries with reasons in the final result.
- **No retries**: Retry logic is out of scope; timeout means the sub-query is skipped.

## Phases

### Phase 1: Task Registry & Cancellation [P0]

**Purpose**: Establish task tracking infrastructure and enable graceful cancellation with proper state persistence.

**Files**:
- `src/foundry_mcp/core/task_registry.py` (new)
- `src/foundry_mcp/core/background_task.py`
- `src/foundry_mcp/workflows/deep_research_workflow.py`

**Tasks**:
1. Create `TaskRegistry` singleton with `Dict[str, BackgroundTask]` storage
2. Add `asyncio.Lock` for thread-safe registry operations
3. Implement `register(task)`, `get(task_id)`, `remove(task_id)` methods
4. Add cleanup task: remove tasks 5 minutes after terminal state
5. Add `cancellation_event: asyncio.Event` to `BackgroundTask`
6. Implement two-phase `BackgroundTask.cancel()`: set event, wait 5s, then force cancel
7. Add cancellation checks at iteration boundaries in `_execute_phase()`
8. Add cancellation check before each provider call in the workflow
9. Catch `CancelledError` at workflow top-level, persist `cancelling` -> cleanup -> `cancelled`
10. Add `try/finally` blocks for resource cleanup (connections, temp files)
11. Implement partial result policy: discard incomplete iteration results
12. **Test**: Unit test verifies task registry add/get/remove with concurrency
13. **Test**: Unit test cancels mid-phase, verify state transitions: running -> cancelling -> cancelled
14. **Test**: Unit test verifies partial results from interrupted iteration are discarded
15. **Test**: Integration test confirms cooperative cancellation completes within 5s (mock provider)
16. **Test**: Integration test confirms forced cancellation completes within 10s (mock blocking provider)

**Verification**: All tests pass; cancellation persists correct state transitions and cleans up resources.

---

### Phase 2: Timeout Watchdog [P0]

**Purpose**: Actively detect and handle timed-out background tasks with proper state persistence and audit events.

**Files**:
- `src/foundry_mcp/core/background_task.py`
- `src/foundry_mcp/core/timeout_watchdog.py` (new)
- Status polling endpoints

**Tasks**:
1. Create `TimeoutWatchdog` class with background asyncio task
2. Implement polling loop: check all running tasks every 10 seconds
3. Add `is_timed_out` property to `BackgroundTask`: True if elapsed > task_timeout
4. Add `is_stale` property: True if no heartbeat for >90 seconds while status is `running`
5. On timeout detected: initiate two-phase cancellation, terminal state `timed_out`
6. Persist timeout metadata (`timed_out_at`, `elapsed_seconds`) to state
7. Emit `task.timeout` audit event with task_id, elapsed time, configured timeout
8. Add staleness detection: mark as `stale` in status (informational, not terminal)
9. Watchdog lifecycle: start on app startup, cancel on shutdown
10. Ensure status response includes timeout/staleness metadata when applicable
11. **Test**: Unit test with short timeout verifies `is_timed_out=True` and state transition
12. **Test**: Unit test verifies staleness detection when heartbeat stops for 90s
13. **Test**: Unit test verifies watchdog polling interval (mock time)
14. **Test**: Verify status response includes timeout metadata

**Verification**: All tests pass; timeout actively detected and handled; staleness reported.

---

### Phase 3: Provider Timeout Enforcement [P0]

**Purpose**: Ensure all providers honor request timeouts and fail fast with appropriate errors.

**Files**:
- `src/foundry_mcp/providers/anthropic.py`
- `src/foundry_mcp/providers/openai.py`
- `src/foundry_mcp/providers/ollama.py`
- `src/foundry_mcp/providers/google.py`
- `src/foundry_mcp/providers/tavily.py`
- `src/foundry_mcp/core/exceptions.py` (add `ProviderTimeoutError`)

**Timeout enforcement pattern**:
- Primary: Client-level timeout (httpx `timeout=` param) to properly close connections
- Secondary: `asyncio.wait_for()` as safety net for non-HTTP operations
- On timeout: Raise `ProviderTimeoutError` with provider name, elapsed time, configured timeout

**Tasks**:
1. Add `ProviderTimeoutError(ProviderError)` to `core/exceptions.py` with fields: provider, elapsed, timeout
2. Review anthropic.py `generate()`: verify httpx client timeout, add `asyncio.wait_for()` wrapper, ensure connection cleanup
3. Update anthropic.py per enforcement pattern
4. Review and update openai.py with same pattern
5. Review and update ollama.py with same pattern
6. Review and update google.py with same pattern
7. Review and update tavily.py with same pattern
8. Add workflow-level handling: catch `ProviderTimeoutError` at sub-query level, log, mark failed, continue
9. Include failed sub-queries in final result
10. **Test**: Per-provider unit test with mock 2s delay, 1s timeout, verify `ProviderTimeoutError` raised
11. **Test**: Verify timeout error includes provider name, elapsed time, configured timeout
12. **Test**: Verify workflow continues after provider timeout, final result lists failed sub-queries

**Verification**: All provider timeout tests pass; workflow handles timeouts gracefully.

---

### Phase 4: Progress Heartbeats [P1]

**Purpose**: Provide visibility into long-running task progress through audit events and persisted state.

**Files**:
- `src/foundry_mcp/workflows/deep_research_workflow.py`
- State persistence layer

**Heartbeat specification**:
- Update `last_heartbeat_at` before each significant operation (not after, to handle long calls)
- Update points: phase start, before each provider call, iteration boundaries

**Metric naming convention**: `foundry_mcp_<domain>_<metric>_<unit>` per observability doc

**Tasks**:
1. Emit `phase.started` audit event: `{phase_name, iteration, task_id, timestamp}`
2. Emit `phase.completed` audit event: `{phase_name, iteration, task_id, duration_ms, timestamp}`
3. Emit `llm.call.started` audit event: `{provider, task_id, timestamp}`
4. Emit `llm.call.completed` audit event: `{provider, task_id, duration_ms, status, timestamp}`
5. Update `last_heartbeat_at` BEFORE each provider call (not after)
6. Persist interim state: `current_phase`, `current_iteration`, `last_heartbeat_at`
7. Add metric `foundry_mcp_research_phase_duration_seconds` (histogram, labels: `phase_name`, `status`)
8. Add metric `foundry_mcp_research_llm_call_duration_seconds` (histogram, labels: `provider`, `status`)
9. **Test**: Verify status response shows `current_phase` and `last_heartbeat_at` during execution
10. **Test**: Verify audit log contains phase and LLM call events with correct structure
11. **Test**: Verify heartbeat updated before (not after) provider call

**Verification**: Status response shows progress; metrics and audit events emitted correctly.

---

### Phase 5: Default Timeout Configuration [P1]

**Purpose**: Provide sensible default timeouts via configuration with clear precedence rules.

**Files**:
- `src/foundry_mcp/routers/research.py` (deep_research action handler)
- `src/foundry_mcp/core/config.py` (config schema)
- `src/foundry_mcp/core/constants.py` (add `DEFAULT_DEEP_RESEARCH_TIMEOUT = 600`)
- `foundry-mcp.toml` (config file)

**Precedence**: explicit param > config file > hardcoded fallback (600s)

**Tasks**:
1. Add `DEFAULT_DEEP_RESEARCH_TIMEOUT = 600` constant to `core/constants.py`
2. Add `research.deep_research_timeout` to config schema (default: 600)
3. Update config model to include the new field
4. Verify config loader handles missing field gracefully (backward compatible)
5. Apply default in action handler when `task_timeout` not explicitly set
6. Document precedence in code comments
7. Expose effective timeout in tool metadata response
8. **Test**: Config default applies when param omitted
9. **Test**: Explicit param overrides config
10. **Test**: Hardcoded fallback applies when config section missing

**Verification**: Tests confirm precedence behavior; tool metadata shows effective timeout.

**Depends on**: Phase 2 (watchdog must exist to enforce timeout)

---

### Phase 6: Executor Isolation [P2]

**Purpose**: Isolate blocking provider operations in a dedicated executor to prevent starvation.

**Risk**: Thread/process pool starvation, deadlock if misconfigured. May be unnecessary if all operations are async.

**Prerequisite**: Audit identifies blocking operations that justify this phase.

**Files**:
- `src/foundry_mcp/providers/provider_manager.py`
- `src/foundry_mcp/core/executor.py` (new module)
- `src/foundry_mcp/core/config.py`

**Lifecycle**:
- Executor created on application startup
- Graceful shutdown with 30s drain timeout on application termination
- If drain timeout exceeded, log warning and force shutdown

**Fallback behavior**:
- Triggered when: queue depth exceeds `executor_queue_limit`
- Action: Log warning with metric increment, route to shared executor
- If both exhausted: Raise `ExecutorExhaustedError`, fail the request
- Alerting: If fallback rate >10% over 1 minute, emit alert-level log

**Tasks**:
1. **Audit**: Identify blocking operations in all providers (sync file I/O, CPU-bound parsing, etc.)
2. Document which operations require executor isolation (if any); if none, mark phase as skipped
3. Create `executor.py` module with `ProviderExecutor` class wrapping `ThreadPoolExecutor`
4. Add config: `providers.executor_pool_size` (default: 4)
5. Add config: `providers.executor_queue_limit` (default: 16)
6. Tie executor lifecycle to application startup/shutdown hooks
7. Implement graceful shutdown with 30s drain timeout
8. Route identified blocking operations through dedicated executor
9. Implement fallback to shared executor when queue limit reached
10. Add `ExecutorExhaustedError` for when both executors unavailable
11. Add metric `foundry_mcp_executor_active_workers` (gauge)
12. Add metric `foundry_mcp_executor_queued_tasks` (gauge)
13. Add metric `foundry_mcp_executor_fallback_total` (counter)
14. Add feature flag `providers.executor_isolation_enabled` (default: true) for rollback
15. **Test**: Load test with 10 concurrent sessions, simulated 500ms blocking operation:
    - All requests complete within 2x expected duration
    - No `ExecutorExhaustedError` raised
    - Executor metrics show utilization
16. **Test**: Verify graceful shutdown completes within drain timeout
17. **Test**: Verify feature flag disables executor routing

**Verification**: Audit documented; load test passes (if applicable); feature flag works.

---

### Phase 7: Test Integration & Coverage [P0]

**Purpose**: Verify cross-cutting test coverage and fixture consistency after all implementation phases.

**Note**: Per-feature tests are included in Phases 1-6. This phase focuses on integration and coverage verification.

**Files**:
- `tests/` directory
- Fixture files

**Tasks**:
1. Run full test suite, verify no regressions from phases 1-6
2. Add integration test: full deep research flow with cancellation mid-way
3. Add integration test: full deep research flow with timeout trigger
4. Add integration test: crash recovery from `cancelling` state
5. Update response fixtures if schemas changed (timeout metadata, staleness, failed sub-queries)
6. Verify all new code paths have >80% coverage (run coverage report)
7. Fix any coverage gaps identified
8. Verify `make test` passes with all new tests

**Verification**: Full test suite passes; coverage threshold met; fixtures consistent.

**Depends on**: Phases 1, 2, 3, 4, 5, 6 (all implementation phases)

---

### Phase 8: Documentation [P1]

**Purpose**: Document new behavior, configuration options, and troubleshooting guidance.

**Files**:
- `docs/troubleshooting.md`
- `docs/deep-research.md`
- `CHANGELOG.md`
- Tool metadata

**Tasks**:
1. Document timeout behavior: task-level vs provider-level, precedence rules
2. Document cancellation behavior: two-phase model, cleanup guarantees, state transitions
3. Document heartbeat/staleness detection behavior (informational vs terminal)
4. Document crash recovery: handling of `cancelling` state on restart
5. Add troubleshooting: "Task timed out" — causes, config tuning, debugging
6. Add troubleshooting: "Task cancelled" — expected behavior, state recovery
7. Add troubleshooting: "Task stale" — detection, causes (long provider calls), when to worry
8. Document new config options with defaults and examples:
   - `research.deep_research_timeout`
   - `providers.executor_pool_size` (if Phase 6 implemented)
   - `providers.executor_queue_limit` (if Phase 6 implemented)
   - `providers.executor_isolation_enabled` (if Phase 6 implemented)
9. Update tool descriptions with timeout parameter semantics
10. Update CHANGELOG.md with new resilience features
11. Verify docs build without warnings

**Verification**: Docs build cleanly; all new config and behavior documented; changelog updated.

**Depends on**: Phases 1, 2, 3, 4, 5, 6

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Executor pool starvation under load | High | Fallback to shared executor; configurable pool size; monitoring metrics; feature flag to disable |
| Deadlock in cancellation paths | High | Two-phase cancellation with timeout; asyncio patterns; no locks during cleanup |
| Connection leaks on timeout | Medium | Client-level timeouts as primary; explicit connection cleanup in finally blocks |
| Incomplete provider timeout coverage | Medium | Audit all providers; per-provider unit tests |
| State corruption on crash during cancellation | Medium | Atomic state transitions (cancelling -> cancelled); crash recovery procedure |
| False staleness during long provider calls | Low | Update heartbeat BEFORE call; 90s threshold (3x interval) |

## Feature Flags (for rollback)

| Flag | Default | Purpose |
|------|---------|---------|
| `providers.executor_isolation_enabled` | true | Disable executor isolation if issues emerge |

## Success Criteria

- [ ] Task registry tracks all background tasks with thread-safe operations
- [ ] Two-phase cancellation: cooperative (5s) then forced; proper state transitions
- [ ] Crash recovery handles `cancelling` state correctly
- [ ] Timeout watchdog actively monitors and cancels timed-out tasks
- [ ] Staleness detection reports tasks with stopped heartbeats (informational)
- [ ] All providers honor request.timeout and raise ProviderTimeoutError
- [ ] Workflow handles provider timeouts: logs, marks failed, continues, reports
- [ ] No connection leaks under timeout-heavy workloads
- [ ] Progress heartbeats visible in status polling and audit logs
- [ ] Default timeout applied from config with clear precedence
- [ ] Executor isolation prevents starvation (if blocking operations exist)
- [ ] Feature flag allows disabling executor isolation
- [ ] All new code paths have >80% test coverage
- [ ] Documentation complete for all new behavior and config options
- [ ] CHANGELOG updated with new features
