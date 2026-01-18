# Agent Operating Guide

This repository follows the MCP Tool Industry Best Practices located under [dev_docs/mcp_best_practices](dev_docs/mcp_best_practices). Treat this file as the root-level contract for how to work inside `foundry-mcp`.

## Session Kickoff

1. **Always skim [dev_docs/mcp_best_practices/README.md#L24-L77](dev_docs/mcp_best_practices/README.md#L24-L77)** when starting a session. This provides the index of authoritative guidance and the code-review checklist you must use before any PR.
2. Map your planned edits to the decision matrix below and re-open the referenced documents _before_ modifying code, specs, or tests.

## Best-Practice Decision Matrix

| Workstream | Required Reads | Non-Negotiable Actions |
|------------|----------------|------------------------|
| New tools, features, or contract tweaks | [dev_docs/codebase_standards/mcp_response_schema.md](dev_docs/codebase_standards/mcp_response_schema.md) (canonical response contract), [dev_docs/mcp_best_practices/01-versioned-contracts.md#L9-L127](dev_docs/mcp_best_practices/01-versioned-contracts.md#L9-L127), [dev_docs/mcp_best_practices/02-envelopes-metadata.md#L30-L285](dev_docs/mcp_best_practices/02-envelopes-metadata.md#L30-L285), [dev_docs/mcp_best_practices/03-serialization-helpers.md#L29-L220](dev_docs/mcp_best_practices/03-serialization-helpers.md#L29-L220), [dev_docs/mcp_best_practices/04-validation-input-hygiene.md#L10-L223](dev_docs/mcp_best_practices/04-validation-input-hygiene.md#L10-L223), [dev_docs/mcp_best_practices/08-security-trust-boundaries.md#L10-L345](dev_docs/mcp_best_practices/08-security-trust-boundaries.md#L10-L345) | Tag every response with schema versions, build envelopes via helpers, validate inputs up front, and enforce security/rate limits before business logic. Document any contract change in specs and PR notes. |
| Pagination, streaming, batching, or bulk results | [dev_docs/mcp_best_practices/06-pagination-streaming.md#L32-L332](dev_docs/mcp_best_practices/06-pagination-streaming.md#L32-L332) and [dev_docs/mcp_best_practices/07-error-semantics.md#L11-L274](dev_docs/mcp_best_practices/07-error-semantics.md#L11-L274) | Use cursor-based pagination, mark partial results, expose `meta.pagination`, and emit actionable errors/warnings following the standard structure. |
| Observability, telemetry, resilience, or timeout logic | [dev_docs/mcp_best_practices/05-observability-telemetry.md#L12-L333](dev_docs/mcp_best_practices/05-observability-telemetry.md#L12-L333) and [dev_docs/mcp_best_practices/12-timeout-resilience.md#L12-L398](dev_docs/mcp_best_practices/12-timeout-resilience.md#L12-L398) | Prefer structured logging, propagate correlation IDs, add metrics/tracing hooks, document timeout budgets, and justify retries/circuit breakers. |
| Feature flags, staged rollouts, or gated experiments | [dev_docs/mcp_best_practices/14-feature-flags.md#L10-L330](dev_docs/mcp_best_practices/14-feature-flags.md#L10-L330) | Document flag lifecycle phases, expose flag status via capabilities, keep override controls ready for rollback, and test both enabled/disabled paths before shipping. |
| Concurrency, async orchestration, or parallel workloads | [dev_docs/mcp_best_practices/15-concurrency-patterns.md#L10-L420](dev_docs/mcp_best_practices/15-concurrency-patterns.md#L10-L420) and [dev_docs/mcp_best_practices/12-timeout-resilience.md#L12-L398](dev_docs/mcp_best_practices/12-timeout-resilience.md#L12-L398) | Declare sync vs async execution in tool docs, enforce concurrency/rate limits, propagate cancellation, and avoid blocking the event loop—fall back to thread/process pools for CPU work. |
| AI-/LLM-facing ergonomics, tool metadata, or discovery flows | [dev_docs/mcp_best_practices/11-ai-llm-integration.md#L11-L352](dev_docs/mcp_best_practices/11-ai-llm-integration.md#L11-L352), [dev_docs/mcp_best_practices/13-tool-discovery.md#L12-L505](dev_docs/mcp_best_practices/13-tool-discovery.md#L12-L505), and [dev_docs/codebase_standards/naming-conventions.md](dev_docs/codebase_standards/naming-conventions.md) | Keep responses concise and structured, design for tool chaining, keep tool descriptions/schemas updated with usage examples, tags, and rate limits, and enforce the canonical prefixes from the naming conventions guide. |
| CLI runtime, naming, or output changes | [dev_docs/cli_best_practices/README.md](dev_docs/cli_best_practices/README.md), [dev_docs/codebase_standards/cli-output.md](dev_docs/codebase_standards/cli-output.md) | Follow JSON-first output, reuse shared runtime helpers, and document namespace/prefix decisions alongside MCP counterparts. |

> **Rule:** If more than one workstream applies, read every referenced section in full before touching files.

## Specs, Docs, and Tests Must Move Together

- Follow [dev_docs/mcp_best_practices/09-spec-driven-development.md#L11-L214](dev_docs/mcp_best_practices/09-spec-driven-development.md#L11-L214): specs are the source of truth. Update specs, implementation, and documentation in the *same* PR, including changelog entries.
- Align fixtures/tests with [dev_docs/mcp_best_practices/10-testing-fixtures.md#L10-L335](dev_docs/mcp_best_practices/10-testing-fixtures.md#L10-L335). Regenerate fixtures whenever schema versions or helpers change, and re-run the freshness checks described there.
- For running tests, debugging failures, and understanding the test suite structure, see [dev_docs/guides/testing.md](dev_docs/guides/testing.md).

## Mandatory Review Checklist

Before opening a PR or merging:
1. Walk through the checklist in [dev_docs/mcp_best_practices/README.md#L64-L70](dev_docs/mcp_best_practices/README.md#L64-L70).
2. Verify every modified tool:
   - Uses standardized envelopes/helpers ([dev_docs/codebase_standards/mcp_response_schema.md](dev_docs/codebase_standards/mcp_response_schema.md), [dev_docs/mcp_best_practices/02-envelopes-metadata.md](dev_docs/mcp_best_practices/02-envelopes-metadata.md), [dev_docs/mcp_best_practices/03-serialization-helpers.md](dev_docs/mcp_best_practices/03-serialization-helpers.md)).
   - Performs validation and sanitation ([dev_docs/mcp_best_practices/04-validation-input-hygiene.md](dev_docs/mcp_best_practices/04-validation-input-hygiene.md), [dev_docs/mcp_best_practices/08-security-trust-boundaries.md](dev_docs/mcp_best_practices/08-security-trust-boundaries.md)).
   - Emits graceful errors/warnings ([dev_docs/mcp_best_practices/07-error-semantics.md](dev_docs/mcp_best_practices/07-error-semantics.md)).
   - Includes observability/resilience hooks if applicable ([dev_docs/mcp_best_practices/05-observability-telemetry.md](dev_docs/mcp_best_practices/05-observability-telemetry.md), [dev_docs/mcp_best_practices/12-timeout-resilience.md](dev_docs/mcp_best_practices/12-timeout-resilience.md)).
   - Documents feature-flag status, capability exposure, and cleanup plans when relevant ([dev_docs/mcp_best_practices/14-feature-flags.md](dev_docs/mcp_best_practices/14-feature-flags.md)).
   - Declares execution model, enforces concurrency/rate limits, and propagates cancellation ([dev_docs/mcp_best_practices/15-concurrency-patterns.md](dev_docs/mcp_best_practices/15-concurrency-patterns.md)).
   - Keeps specs, docs, and fixtures in sync ([dev_docs/mcp_best_practices/09-spec-driven-development.md](dev_docs/mcp_best_practices/09-spec-driven-development.md), [dev_docs/mcp_best_practices/10-testing-fixtures.md](dev_docs/mcp_best_practices/10-testing-fixtures.md)).
3. Mention in the PR description which best-practices sections you consulted.

## When in Doubt

- If code touches security, privacy, or external boundaries, re-read [dev_docs/mcp_best_practices/04-validation-input-hygiene.md](dev_docs/mcp_best_practices/04-validation-input-hygiene.md), [dev_docs/mcp_best_practices/07-error-semantics.md](dev_docs/mcp_best_practices/07-error-semantics.md), [dev_docs/mcp_best_practices/08-security-trust-boundaries.md](dev_docs/mcp_best_practices/08-security-trust-boundaries.md), and [dev_docs/mcp_best_practices/11-ai-llm-integration.md](dev_docs/mcp_best_practices/11-ai-llm-integration.md) entirely and document the applied guidance in commit/PR notes.
- If you are gating behavior behind feature flags or changing concurrency/async behavior, re-read [dev_docs/mcp_best_practices/14-feature-flags.md](dev_docs/mcp_best_practices/14-feature-flags.md) and [dev_docs/mcp_best_practices/15-concurrency-patterns.md](dev_docs/mcp_best_practices/15-concurrency-patterns.md) along with [dev_docs/mcp_best_practices/12-timeout-resilience.md](dev_docs/mcp_best_practices/12-timeout-resilience.md) for resilience, and call out the safeguards you applied.
- When ambiguity exists, default to re-reading the full relevant document(s) instead of relying on summaries.
- Record consulted sections in your plan or worklog so reviewers know which guidance informed the change.
