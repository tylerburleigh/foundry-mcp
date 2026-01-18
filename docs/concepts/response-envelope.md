# Response Envelope Guide

All foundry-mcp tools and CLI commands return responses in a standard envelope format called "response-v2".

## Standard Envelope

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "version": "response-v2",
    "request_id": "req_abc123",
    "warnings": [],
    "pagination": { ... },
    "telemetry": { ... }
  }
}
```

## Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the operation completed successfully |
| `data` | object | Operation payload (empty `{}` when no data) |
| `error` | string/null | Error message when `success` is `false` |
| `meta` | object | Operational metadata |

---

## Success Response

When an operation succeeds:

```json
{
  "success": true,
  "data": {
    "spec_id": "my-feature-2025-01-15-001",
    "title": "My Feature",
    "status": "active",
    "progress": 0.45
  },
  "error": null,
  "meta": {
    "version": "response-v2",
    "request_id": "req_abc123"
  }
}
```

### Empty Results

Even with no results, the structure stays the same:

```json
{
  "success": true,
  "data": {
    "specs": [],
    "total_count": 0
  },
  "error": null,
  "meta": { "version": "response-v2" }
}
```

---

## Error Response

When an operation fails:

```json
{
  "success": false,
  "data": {
    "error_code": "SPEC_NOT_FOUND",
    "error_type": "not_found",
    "remediation": "Verify the spec ID using 'specs list'",
    "details": {
      "spec_id": "nonexistent-spec"
    }
  },
  "error": "Specification not found: nonexistent-spec",
  "meta": {
    "version": "response-v2",
    "request_id": "req_xyz789"
  }
}
```

### Error Data Fields

| Field | Description |
|-------|-------------|
| `error_code` | Machine-readable code (e.g., `VALIDATION_ERROR`) |
| `error_type` | Error category (e.g., `validation`, `not_found`) |
| `remediation` | How to fix the error |
| `details` | Additional context |

See [Error Codes Reference](../reference/error-codes.md) for all error codes.

---

## Metadata Fields

### version (required)

Always `"response-v2"`. Identifies the envelope schema version.

### request_id (recommended)

Correlation ID for tracing and debugging:

```json
{
  "meta": {
    "request_id": "req_abc123"
  }
}
```

### warnings (optional)

Non-fatal issues that didn't prevent success:

```json
{
  "success": true,
  "data": { "result": "ok" },
  "meta": {
    "warnings": [
      "Deprecated parameter 'old_param' used, use 'new_param' instead",
      "Some optional features unavailable"
    ]
  }
}
```

### pagination (optional)

Cursor-based pagination for list operations:

```json
{
  "meta": {
    "pagination": {
      "cursor": "eyJvZmZzZXQiOjUwfQ==",
      "has_more": true,
      "total_count": 150,
      "page_size": 50
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `cursor` | Opaque token for next page |
| `has_more` | Whether more results exist |
| `total_count` | Total items (if available) |
| `page_size` | Items per page |

**Using pagination:**

```bash
# First request
foundry-cli specs find --limit 50

# Next page (pass cursor from previous response)
foundry-cli specs find --limit 50 --cursor "eyJvZmZzZXQiOjUwfQ=="
```

### rate_limit (optional)

Rate limiting information:

```json
{
  "meta": {
    "rate_limit": {
      "limit": 100,
      "remaining": 45,
      "reset_at": "2025-01-15T10:30:00Z"
    }
  }
}
```

### telemetry (optional)

Performance metrics:

```json
{
  "meta": {
    "telemetry": {
      "duration_ms": 125.4,
      "db_queries": 3,
      "cache_hit": true
    }
  }
}
```

---

## Working with Responses

### CLI: Using jq

```bash
# Extract data
foundry-cli specs find | jq '.data.specs'

# Check success
foundry-cli validate check my-spec | jq '.success'

# Get error message
foundry-cli tasks complete bad-spec bad-task | jq '.error'

# Extract with pagination
foundry-cli specs find | jq '.meta.pagination.has_more'
```

### Scripting Example

```bash
#!/bin/bash

result=$(foundry-cli specs find --status active)
success=$(echo "$result" | jq -r '.success')

if [ "$success" = "true" ]; then
    specs=$(echo "$result" | jq -r '.data.specs[].id')
    for spec in $specs; do
        echo "Processing: $spec"
    done
else
    error=$(echo "$result" | jq -r '.error')
    echo "Error: $error"
    exit 1
fi
```

### Python Example

```python
import subprocess
import json

def call_foundry(command: list[str]) -> dict:
    result = subprocess.run(
        ["foundry-cli"] + command,
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

response = call_foundry(["specs", "find", "--status", "active"])

if response["success"]:
    for spec in response["data"]["specs"]:
        print(f"Spec: {spec['id']} - {spec['title']}")

    if response["meta"].get("pagination", {}).get("has_more"):
        cursor = response["meta"]["pagination"]["cursor"]
        # Fetch next page...
else:
    error = response["error"]
    remediation = response["data"].get("remediation", "")
    print(f"Error: {error}")
    print(f"Fix: {remediation}")
```

---

## Multi-Payload Responses

Some operations return multiple data types:

```json
{
  "success": true,
  "data": {
    "spec": {
      "id": "my-spec",
      "title": "My Spec"
    },
    "tasks": [
      { "id": "task-1", "status": "pending" },
      { "id": "task-2", "status": "completed" }
    ],
    "progress": {
      "completed": 1,
      "total": 2,
      "percentage": 50
    }
  },
  "meta": { "version": "response-v2" }
}
```

---

## Edge Cases

### Partial Success

Some operations may partially succeed:

```json
{
  "success": true,
  "data": {
    "processed": 8,
    "failed": 2,
    "results": [ ... ]
  },
  "meta": {
    "warnings": [
      "2 items failed processing",
      "See 'failed_items' in data for details"
    ]
  }
}
```

### Blocked Operations

When work is blocked but not an error:

```json
{
  "success": true,
  "data": {
    "task_id": "task-1-2",
    "status": "blocked",
    "blocked_by": ["task-1-1"],
    "can_start": false
  },
  "meta": {
    "warnings": ["Task blocked by incomplete dependencies"]
  }
}
```

---

## Related

- [Error Codes Reference](../reference/error-codes.md) - All error codes
- [MCP Tool Reference](../05-mcp-tool-reference.md) - Tool documentation
- [CLI Command Reference](../04-cli-command-reference.md) - CLI documentation
