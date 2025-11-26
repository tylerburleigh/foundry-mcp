# MCP Response Schema Guide

This document captures the canonical response contract that every Foundry MCP operation must follow. It complements the completed specification [`specs/completed/response-schema-standardization-2025-11-26-001.json`](../../specs/completed/response-schema-standardization-2025-11-26-001.json), the shared helpers in [`src/foundry_mcp/core/responses.py`](../../src/foundry_mcp/core/responses.py), and the best-practice guidance in [docs/mcp_best_practices](../mcp_best_practices/README.md).

## Standard Envelope

All tool responses **must** serialize to the following structure ("response-v2"):

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "version": "response-v2",
    "request_id": "req_abc123"?,
    "warnings": ["..."]?,
    "pagination": { ... }?,
    "rate_limit": { ... }?,
    "telemetry": { ... }?
  }
}
```

- `success` (bool): Indicates whether the operation completed successfully.
- `data` (object): Operation payload. When no payload exists, send `{}`.
- `error` (string | null): Populated only when `success` is `false`.
- `meta` (object): Always include `{"version": "response-v2"}` and attach optional metadata using the reserved keys listed below.

### Metadata Field Semantics

| Key              | Required | Description |
|------------------|----------|-------------|
| `version`        | YES      | Identifies the response schema version (`response-v2`). |
| `request_id`     | SHOULD   | Correlation identifier propagated through logs/traces. |
| `warnings`       | SHOULD   | Non-fatal issues for successful operations (array of strings). |
| `pagination`     | MAY      | Cursor-based pagination object containing `cursor`, `has_more`, `total_count`, etc. |
| `rate_limit`     | MAY      | Remaining quota, reset time, and retry hints when throttling occurs. |
| `telemetry`      | MAY      | Timing/performance metrics such as `duration_ms` or downstream call counts. |

> Do **not** invent new top-level keys under `data` to convey metadata. Attach operational context through `meta` so every tool shares the same envelope semantics.

### Edge Case Semantics

| Scenario                    | Required Behavior |
|----------------------------|--------------------|
| Empty but successful query | `success: true`, include empty arrays/counts in `data`, `error: null`. |
| Missing resource / invalid input | `success: false`, `data` contains structured error info, descriptive `error` string. |
| Blocked or partial work    | `success: true`, describe state inside `data`, add `meta.warnings` if applicable. |
| Multi-payload operations   | Nest each payload under a named key inside `data` (e.g., `{ "spec": {...}, "tasks": [...] }`). |

## Helper Usage

Always leverage the shared helpers and dataclasses to produce responses:

```python
from dataclasses import asdict
from foundry_mcp.core.responses import success_response, error_response

@mcp.tool()
def foundry_example(...) -> dict:
    payload = compute_payload(...)
    return asdict(success_response(
        data={"result": payload},
        warnings=payload.warnings,
        pagination=payload.pagination,
        request_id=context.request_id,
    ))
```

For failures, include machine-readable context:

```python
return asdict(error_response(
    "Validation failed",
    error_code="INVALID_INPUT",
    error_type="validation",
    remediation="Provide a non-empty spec_id",
    data={"field": "spec_id"},
    request_id=context.request_id,
    rate_limit=rate_limit_state,
))
```

These helpers guarantee `meta.version` is present and prevent ad-hoc response shapes. Avoid constructing dicts manually.

## Implementation Checklist

1. **Import helpers** (`success_response` / `error_response`) in every tool module.
2. **Return `asdict(...)`** so dataclasses serialize with the standardized keys.
3. **Keep `data` payloads business-focused**; put operational context in `meta`.
4. **Document any additional `meta` semantics** (new pagination fields, telemetry) in tool specs.
5. **Record deviations or streaming quirks** in specs to prevent regressions.

## Testing Requirements

- Unit enforcement lives in [`tests/test_responses.py`](../../tests/test_responses.py); extend it when updating helpers.
- Integration tests such as [`tests/integration/test_mcp_tools.py`](../../tests/integration/test_mcp_tools.py) should assert the envelope for new tools.
- Fixtures and parity harnesses must verify `meta.version == "response-v2"` and any declared metadata keys (`warnings`, `pagination`, etc.).

## Compatibility & Rollout Notes

- The `response_contract_v2` feature flag governs client opt-in. During migrations, continue returning v1 only when explicitly required and document timelines in the spec.
- Feature-flag lifecycles must follow [docs/mcp_best_practices/14-feature-flags.md](../mcp_best_practices/14-feature-flags.md), and metadata such as rate limits should align with [docs/mcp_best_practices/02-envelopes-metadata.md](../mcp_best_practices/02-envelopes-metadata.md).
- Telemetry counters in `foundry_mcp/server.py` rely on consistent envelopes; avoid bypassing the helpers or mutating the serialized dict afterward.

## Related References

- Spec: [`response-schema-standardization-2025-11-26-001`](../../specs/completed/response-schema-standardization-2025-11-26-001.json)
- Helpers: [`src/foundry_mcp/core/responses.py`](../../src/foundry_mcp/core/responses.py)
- Testing/fixtures: [docs/mcp_best_practices/10-testing-fixtures.md](../mcp_best_practices/10-testing-fixtures.md)
- Envelopes & metadata guidance: [docs/mcp_best_practices/02-envelopes-metadata.md](../mcp_best_practices/02-envelopes-metadata.md)
