# MCP Response Schema Guide

This document captures the canonical response contract that every Foundry MCP operation must follow. It complements the completed specification [`specs/completed/response-schema-standardization-2025-11-26-001.json`](../specs/completed/response-schema-standardization-2025-11-26-001.json) and the code-level helpers in [`src/foundry_mcp/core/responses.py`](../src/foundry_mcp/core/responses.py). See also [`docs/mcp_tool_industry_best_practices.md`](./mcp_tool_industry_best_practices.md) for broader ecosystem guidance.

## Standard Envelope

All tool responses **must** serialize to the following structure ("response-v2"):

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "version": "response-v2",
    "warnings": []?,
    "pagination": { ... }?,
    "telemetry": { ... }?
  }
}
```

- `success` (bool): Indicates whether the operation completed successfully.
- `data` (object): Operation payload. When no payload exists, send `{}`.
- `error` (string | null): Populated only when `success` is `false`.
- `meta` (object): Must always include `{"version": "response-v2"}`; may include operation metadata (pagination, rate limits, warnings, etc.).

### Reserved Keys Inside `data`

When a tool needs to attach additional metadata without polluting the root envelope, use these conventions in `data`:

- `_meta`: Per-operation metadata (e.g., pagination cursors, timing information).
- `_warnings`: Non-fatal issues encountered during execution.

### Edge Case Semantics

| Scenario                         | Required Behavior |
|---------------------------------|--------------------|
| Empty but successful query      | `success: true`, include empty arrays/counts in `data`, `error: null`.
| Missing resource / invalid input| `success: false`, `data: {}`, descriptive `error` string.
| Blocked or partial work         | `success: true`, describe blocking state in `data`, optionally set `_warnings`.
| Multi-payload operations        | Nest each payload under a named key inside `data` (e.g., `{"spec": {...}, "tasks": [...]}`).

## Helper Usage

Always leverage the shared helpers and dataclass to produce responses:

```python
from dataclasses import asdict
from foundry_mcp.core.responses import success_response, error_response

@mcp.tool()
def foundry_example(...) -> dict:
    try:
        payload = compute_payload(...)
        return asdict(success_response(**payload))
    except KnownError as exc:
        return asdict(error_response(str(exc)))
```

These helpers guarantee `meta.version` is present and prevent ad-hoc response shapes. Avoid constructing dicts manually.

## Implementation Checklist

1. **Import helpers** (`success_response` / `error_response`) in every new tool module.
2. **Return `asdict(...)`** so dataclasses serialize with the standardized keys.
3. **Populate `data` only with operation-specific fields**; never place raw booleans like `found` at the top level of the envelope.
4. **Document `meta` semantics** (e.g., new pagination keys) in both the tool docstring and relevant specs.
5. **Capture deviations** (streaming, multi-payload quirks) inside the spec so future refactors do not regress the behavior.

## Testing Requirements

- Unit-level enforcement lives in [`tests/test_responses.py`](../tests/test_responses.py); extend it when updating the helpers.
- Each adapter addition should extend the appropriate suite (typically [`tests/integration/test_mcp_tools.py`](../tests/integration/test_mcp_tools.py)) to assert the envelope for the new tool(s).
- When adding CLI workflows or higher-level commands, ensure any recorded fixtures and parity harnesses assert `meta.version == "response-v2"`.

## Compatibility & Rollout Notes

- The `response_contract_v2` capability flag governs client opt-in. During migrations, continue returning v1 only when explicitly required and document timelines in the relevant spec.
- Telemetry counters in `foundry_mcp/server.py` rely on consistent envelopes; avoid bypassing the helpers or direct dict mutations after serialization.

## References

- Spec: [`response-schema-standardization-2025-11-26-001`](../specs/completed/response-schema-standardization-2025-11-26-001.json)
- Helpers: [`src/foundry_mcp/core/responses.py`](../src/foundry_mcp/core/responses.py)
- Backlog guardrails: [`OPERATIONS_TO_ADD.md`](../OPERATIONS_TO_ADD.md)
