# 1. Stable, Versioned Contracts

> Design schemas that evolve gracefully without breaking clients.

## Overview

Every MCP tool response represents a contract between the server and its clients. Breaking this contract causes failures in production. This section covers how to version schemas and manage their evolution.

## Requirements

### MUST

- **Publish explicit schema versions** for every response contract
- **Include version in responses** via `meta.version` field
- **Maintain backward compatibility** within a major version
- **Document breaking changes** with migration guides

### SHOULD

- **Use semantic versioning** for schema versions (e.g., `response-v2.1.0`)
- **Provide deprecation windows** of at least 2 release cycles before removing fields
- **Support version negotiation** via capability flags or headers

### MAY

- **Support multiple active versions** concurrently during migrations
- **Use date-based versioning** for rapid iteration phases (e.g., `2025-11-26`)

## Versioning Strategies

### Semantic Versioning (Recommended)

```
MAJOR.MINOR.PATCH
  │     │     └── Bug fixes, no contract change
  │     └──────── New optional fields, backward compatible
  └────────────── Breaking changes, new envelope structure
```

**Example:**
```python
# Version progression
"response-v1.0.0"  # Initial release
"response-v1.1.0"  # Added optional `meta.rate_limit` field
"response-v1.1.1"  # Fixed serialization bug
"response-v2.0.0"  # Changed `data` structure (breaking)
```

### Date-Based Versioning

Use when iterating rapidly or for experimental features:

```python
"response-2025-11-26"  # Snapshot version
"response-2025-12-01"  # Next iteration
```

## Version Negotiation

### Capability Flags

```python
# Client requests specific version
capabilities = {"response_contract": "v2"}

# Server checks and responds appropriately
if client_supports_v2(capabilities):
    return v2_response(data)
else:
    return v1_response(data)  # Legacy fallback
```

### Header-Based Negotiation

```http
# Request
Accept-Version: response-v2

# Response
Content-Version: response-v2.1.0
```

## Backward Compatibility Rules

### Safe Changes (Non-Breaking)

- Adding new optional fields to `data`
- Adding new keys to `meta`
- Adding new enum values (if clients handle unknown values)
- Relaxing validation (accepting more input formats)

### Breaking Changes (Require Major Version)

- Removing or renaming fields
- Changing field types
- Adding required fields
- Tightening validation
- Changing envelope structure

## Deprecation Process

```python
# Phase 1: Announce deprecation (keep field, add warning)
{
    "success": true,
    "data": {
        "old_field": "value",      # Deprecated
        "new_field": "value"       # Replacement
    },
    "meta": {
        "version": "response-v1.2.0",
        "warnings": ["'old_field' is deprecated, use 'new_field'. Removal in v2.0.0"]
    }
}

# Phase 2: Remove in next major version
{
    "success": true,
    "data": {
        "new_field": "value"
    },
    "meta": {
        "version": "response-v2.0.0"
    }
}
```

## Documenting Experimental Features

When introducing unstable features, mark them clearly:

```python
{
    "data": {
        "result": "...",
        "_experimental_streaming": {
            "enabled": true,
            "chunk_size": 1024
        }
    },
    "meta": {
        "version": "response-v2.0.0",
        "warnings": ["'_experimental_streaming' is unstable and may change"]
    }
}
```

## Anti-Patterns

### Don't: Implicit Versioning

```python
# Bad: No version indicator
{"success": true, "data": {...}}

# Good: Explicit version
{"success": true, "data": {...}, "meta": {"version": "response-v2"}}
```

### Don't: Breaking Changes Without Version Bump

```python
# Bad: Changed structure, same version
# v1: {"data": {"items": [...]}}
# v1: {"data": {"results": [...]}}  # Renamed field!

# Good: Version bump for breaking change
# v1: {"data": {"items": [...]}, "meta": {"version": "v1"}}
# v2: {"data": {"results": [...]}, "meta": {"version": "v2"}}
```

## Related Documents

- [Envelopes & Metadata](./02-envelopes-metadata.md) - Response structure standards
- [Spec-Driven Development](./09-spec-driven-development.md) - Documenting contracts
- [Response Schema Reference](../codebase_standards/mcp_response_schema.md) - Canonical contract

---

**Navigation:** [Index](./README.md) | [Next: Envelopes & Metadata →](./02-envelopes-metadata.md)
