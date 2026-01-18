# 2. Consistent Envelopes & Metadata

> Wrap all responses in predictable structures so clients don't need tool-specific logic.

## Overview

A consistent response envelope allows clients to handle success, errors, and metadata uniformly across all tools. This eliminates special-case handling and simplifies adapter development.

## Requirements

### MUST

- **Use the standard envelope** for all tool responses
- **Include `meta.version`** in every response
- **Keep business data inside `data`** - never at envelope root
- **Set `error` to `null`** on success, descriptive string on failure

### SHOULD

- **Reserve metadata keys** and document their semantics
- **Use `meta.warnings`** for non-fatal issues
- **Include correlation IDs** in `meta` for tracing

### MAY

- **Add pagination info** in `meta.pagination`
- **Include rate limit info** in `meta.rate_limit`
- **Add timing data** in `meta.telemetry`

## Standard Envelope Structure

```json
{
    "success": true,
    "data": { },
    "error": null,
    "meta": {
        "version": "response-v2",
        "request_id": "req_abc123",
        "warnings": [],
        "pagination": null,
        "rate_limit": null,
        "telemetry": null
    }
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `success` | boolean | MUST | `true` if operation completed, `false` on failure |
| `data` | object | MUST | Operation payload; `{}` when empty |
| `error` | string \| null | MUST | Error message when `success=false`, otherwise `null` |
| `meta` | object | MUST | Response metadata |
| `meta.version` | string | MUST | Schema version identifier |
| `meta.request_id` | string | SHOULD | Correlation ID for tracing |
| `meta.warnings` | string[] | MAY | Non-fatal issues encountered |
| `meta.pagination` | object | MAY | Pagination cursors/info |
| `meta.rate_limit` | object | MAY | Rate limit status |
| `meta.telemetry` | object | MAY | Timing/performance data |

## Reserved Metadata Keys

These keys have standardized semantics across all tools:

### `meta.request_id`

Correlation ID for distributed tracing:

```json
{
    "meta": {
        "request_id": "req_a1b2c3d4e5f6",
        "trace_id": "trace_xyz789",
        "span_id": "span_123"
    }
}
```

### `meta.warnings`

Non-fatal issues that didn't prevent success:

```json
{
    "success": true,
    "data": {"results": [...]},
    "meta": {
        "warnings": [
            "3 items skipped due to permission errors",
            "Results truncated to 1000 items"
        ]
    }
}
```

### `meta.pagination`

Cursor-based pagination info:

```json
{
    "meta": {
        "pagination": {
            "cursor": "eyJvZmZzZXQiOjEwMH0=",
            "has_more": true,
            "total_count": 5432,
            "page_size": 100
        }
    }
}
```

### `meta.rate_limit`

Rate limiting status:

```json
{
    "meta": {
        "rate_limit": {
            "limit": 100,
            "remaining": 42,
            "reset_at": "2025-11-26T12:00:00Z",
            "retry_after_seconds": null
        }
    }
}
```

### `meta.telemetry`

Performance metrics:

```json
{
    "meta": {
        "telemetry": {
            "duration_ms": 234,
            "db_queries": 3,
            "cache_hit": true
        }
    }
}
```

## Success Response Examples

### Simple Success

```json
{
    "success": true,
    "data": {
        "message": "Operation completed"
    },
    "error": null,
    "meta": {
        "version": "response-v2"
    }
}
```

### Success with Data

```json
{
    "success": true,
    "data": {
        "user": {
            "id": "usr_123",
            "name": "Alice",
            "email": "alice@example.com"
        }
    },
    "error": null,
    "meta": {
        "version": "response-v2",
        "request_id": "req_abc123"
    }
}
```

### Success with Warnings

```json
{
    "success": true,
    "data": {
        "processed": 97,
        "skipped": 3
    },
    "error": null,
    "meta": {
        "version": "response-v2",
        "warnings": [
            "3 records skipped: invalid format"
        ]
    }
}
```

## Error Response Examples

### Simple Error

```json
{
    "success": false,
    "data": {},
    "error": "Resource not found: user_id 'usr_999' does not exist",
    "meta": {
        "version": "response-v2",
        "request_id": "req_xyz789"
    }
}
```

### Error with Context

```json
{
    "success": false,
    "data": {
        "validation_errors": [
            {"field": "email", "message": "Invalid email format"},
            {"field": "age", "message": "Must be positive integer"}
        ]
    },
    "error": "Validation failed: 2 errors",
    "meta": {
        "version": "response-v2"
    }
}
```

## Anti-Patterns

### Don't: Put Business Data at Root

```python
# Bad: Data at envelope root
{
    "success": true,
    "user_id": "123",      # Should be in data
    "user_name": "Alice",  # Should be in data
    "meta": {...}
}

# Good: Data inside data field
{
    "success": true,
    "data": {
        "user_id": "123",
        "user_name": "Alice"
    },
    "meta": {...}
}
```

### Don't: Inconsistent Error Handling

```python
# Bad: Different error formats
{"success": false, "error": "Not found"}
{"success": false, "message": "Invalid input"}  # Different key!
{"success": false, "errors": ["Error 1", "Error 2"]}  # Array!

# Good: Consistent format
{"success": false, "error": "Not found", "data": {}, "meta": {...}}
{"success": false, "error": "Invalid input", "data": {}, "meta": {...}}
{"success": false, "error": "Multiple errors", "data": {"errors": [...]}, "meta": {...}}
```

### Don't: Omit Version

```python
# Bad: Missing version
{"success": true, "data": {...}}

# Good: Always include version
{"success": true, "data": {...}, "meta": {"version": "response-v2"}}
```

## Related Documents

- [Response Schema Reference](../codebase_standards/mcp_response_schema.md) - Canonical response contract
- [Versioned Contracts](./01-versioned-contracts.md) - Schema versioning
- [Serialization Helpers](./03-serialization-helpers.md) - Helper functions
- [Error Semantics](./07-error-semantics.md) - Error handling patterns

---

**Navigation:** [← Versioned Contracts](./01-versioned-contracts.md) | [Index](./README.md) | [Next: Serialization Helpers →](./03-serialization-helpers.md)
