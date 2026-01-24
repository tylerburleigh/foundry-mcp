# MCP Server Resilience

## Objective

Add comprehensive exception handling to prevent MCP server crashes when unexpected errors occur in research workflows and unified tool dispatch functions. Currently, unhandled exceptions can crash the server; this work ensures graceful error responses are returned instead.

## Scope

### In Scope
- Add top-level try/catch to 4 research workflow execute() methods
- Add catch-all exception handlers to 15 unified tool dispatch functions (16 total exist, but research is already protected)
- Add tests verifying exception handling behavior
- Follow existing patterns from DeepResearchWorkflow and _dispatch_research_action
- Catch `Exception` only (not `BaseException`) to allow `SystemExit`/`KeyboardInterrupt` to propagate
- Note: All workflows and dispatch functions are synchronous; `asyncio.CancelledError` handling is not needed

### Out of Scope
- Refactoring existing error handling logic
- Adding new error types or codes
- Performance optimization
- Documentation updates beyond code comments

## Phases

### Phase 1: Research Workflow Exception Handling

**Purpose**: Protect the 4 research workflows that lack top-level exception handling, preventing crashes during workflow execution.

**Files**:
- `src/foundry_mcp/core/research/workflows/chat.py`
- `src/foundry_mcp/core/research/workflows/thinkdeep.py`
- `src/foundry_mcp/core/research/workflows/consensus.py`
- `src/foundry_mcp/core/research/workflows/ideate.py`

**Tasks**:
1. Add try/catch wrapper to ChatWorkflow.execute() - returns WorkflowResult on exception
2. Add try/catch wrapper to ThinkDeepWorkflow.execute() - returns WorkflowResult on exception
3. Add try/catch wrapper to ConsensusWorkflow.execute() - returns WorkflowResult on exception
4. Add try/catch wrapper to IdeateWorkflow.execute() - returns WorkflowResult on exception

**Pattern** (from DeepResearchWorkflow):
```python
def execute(self, ...) -> WorkflowResult:
    try:
        # existing logic
    except Exception as exc:
        logger.exception("Workflow execute failed: %s", exc)
        return WorkflowResult(
            success=False,
            content="",
            error=f"Workflow failed: {exc}",
            metadata={"workflow": "chat", "error_type": exc.__class__.__name__},
        )
```

**Note**: Workflows return `WorkflowResult` objects. Dispatch functions (Phase 2) wrap these and translate to `dict` via `asdict(error_response(...))`. This is intentional - the dispatch layer normalizes all responses to the MCP response envelope format.

**Verification**: Run `pytest tests/core/research/workflows/` - all tests pass

### Phase 2: Unified Tool Dispatch Exception Handling (15 of 16 functions)

**Purpose**: Protect 15 dispatch functions that only catch ActionRouterError, leaving other exceptions to crash the server.

**Files**:
- `src/foundry_mcp/tools/unified/authoring.py`
- `src/foundry_mcp/tools/unified/environment.py`
- `src/foundry_mcp/tools/unified/error.py`
- `src/foundry_mcp/tools/unified/health.py`
- `src/foundry_mcp/tools/unified/journal.py`
- `src/foundry_mcp/tools/unified/lifecycle.py`
- `src/foundry_mcp/tools/unified/plan.py`
- `src/foundry_mcp/tools/unified/provider.py`
- `src/foundry_mcp/tools/unified/pr.py`
- `src/foundry_mcp/tools/unified/review.py`
- `src/foundry_mcp/tools/unified/server.py`
- `src/foundry_mcp/tools/unified/spec.py`
- `src/foundry_mcp/tools/unified/task.py`
- `src/foundry_mcp/tools/unified/test.py`
- `src/foundry_mcp/tools/unified/verification.py`

**Tasks** (15 dispatch functions - research already has catch-all):
1. Add catch-all to _dispatch_authoring_action (authoring.py)
2. Add catch-all to _dispatch_environment_action (environment.py)
3. Add catch-all to _dispatch_error_action (error.py)
4. Add catch-all to _dispatch_health_action (health.py)
5. Add catch-all to _dispatch_journal_action (journal.py)
6. Add catch-all to _dispatch_lifecycle_action (lifecycle.py)
7. Add catch-all to _dispatch_plan_action (plan.py)
8. Add catch-all to _dispatch_provider_action (provider.py)
9. Add catch-all to _dispatch_pr_action (pr.py)
10. Add catch-all to _dispatch_review_action (review.py)
11. Add catch-all to _dispatch_server_action (server.py)
12. Add catch-all to _dispatch_spec_action (spec.py)
13. Add catch-all to _dispatch_task_action (task.py)
14. Add catch-all to _dispatch_test_action (test.py)
15. Add catch-all to _dispatch_verification_action (verification.py)

**Pattern** (from _dispatch_research_action):
```python
except Exception as exc:
    logger.exception("Action '%s' failed with unexpected error: %s", action, exc)
    return asdict(
        error_response(
            f"Action '{action}' failed: {str(exc) or exc.__class__.__name__}",
            error_code=ErrorCode.INTERNAL_ERROR,
            error_type=ErrorType.INTERNAL,
            remediation="Check configuration and review logs for details.",
            details={"action": action, "error_type": exc.__class__.__name__},
        )
    )
```

**Verification**: Run `pytest tests/tools/unified/` - all tests pass

### Phase 3: Test Coverage

**Purpose**: Add tests to verify exception handling works correctly and prevents crashes.

**Workflow Test Files**:
- `tests/core/research/workflows/test_chat.py`
- `tests/core/research/workflows/test_thinkdeep.py`
- `tests/core/research/workflows/test_consensus.py`
- `tests/core/research/workflows/test_ideate.py`

**Dispatch Test Files** (naming convention: `test_{tool}.py`):
- `tests/tools/unified/test_authoring.py`
- `tests/tools/unified/test_environment.py`
- `tests/tools/unified/test_error.py`
- `tests/tools/unified/test_health.py`
- `tests/tools/unified/test_journal.py`
- `tests/tools/unified/test_lifecycle.py`
- `tests/tools/unified/test_plan.py`
- `tests/tools/unified/test_provider.py`
- `tests/tools/unified/test_pr.py`
- `tests/tools/unified/test_review.py`
- `tests/tools/unified/test_server.py`
- `tests/tools/unified/test_spec.py`
- `tests/tools/unified/test_task.py`
- `tests/tools/unified/test_test.py`
- `tests/tools/unified/test_verification.py`

**Tasks**:
1. Add test_execute_catches_exceptions to each workflow test file (4 tests)
2. Add test_dispatch_exception_returns_error_response to each dispatch test file (15 tests)

**Verification**: Run `pytest -k "catches_exception"` - all new tests pass

## Final Verification

Run comprehensive test suite to confirm no regressions:
```bash
pytest tests/core/research/workflows/ tests/tools/unified/ -v
pytest  # Full test suite
```

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing error handling | Medium | Follow existing patterns exactly; run full test suite |
| Missing edge cases | Low | Comprehensive test coverage for each modified function |
| Log spam from caught exceptions | Low | Use logger.exception() which is already standard |
| Unexpected error masking | Medium | Catch `Exception` only (not `BaseException`); include error details in response |

## Rollback Strategy

If issues arise post-deployment:
1. Revert the commit immediately if test failures or unexpected error masking is observed
2. All changes are additive (wrapping existing code) so revert is safe
3. Monitor logs for `"failed with unexpected error"` messages to detect masked issues

## Success Criteria

- [ ] All 4 research workflow execute() methods have try/catch wrappers
- [ ] All 15 dispatch functions have catch-all exception handlers
- [ ] All existing tests pass (no regressions)
- [ ] Full test suite (`pytest`) passes
- [ ] New exception handling tests pass (19 total: 4 workflow + 15 dispatch)
- [ ] Manual verification (one-time during development): Mock a workflow to raise `RuntimeError`, verify error response returned instead of crash. This is also covered by automated tests in Phase 3.
