# 3. Shared Serialization Helpers

> Centralize envelope creation to ensure consistency and prevent drift.

## Overview

Manual response construction leads to inconsistency and bugs. Centralized helper functions enforce the standard envelope structure and simplify tool development.

## Requirements

### MUST

- **Use shared helpers** (`success_response`, `error_response`) for all responses
- **Never construct envelopes manually** in tool implementations
- **Return serialized output** via `asdict()` or equivalent

### SHOULD

- **Keep helpers side-effect free** for business logic (telemetry decorators are acceptable)
- **Update documentation** when adding new helper capabilities
- **Version helper modules** alongside schema versions

### MAY

- **Add specialized helpers** for common patterns (pagination, streaming)
- **Include telemetry hooks** via decorators

## Core Helpers

### `success_response`

Creates a successful response envelope:

```python
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

@dataclass
class SuccessResponse:
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    error: None = None
    meta: Dict[str, Any] = field(default_factory=lambda: {"version": "response-v2"})

def success_response(
    data: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    pagination: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    **extra_meta
) -> SuccessResponse:
    """Create a standardized success response."""
    meta = {"version": "response-v2"}

    if request_id:
        meta["request_id"] = request_id
    if warnings:
        meta["warnings"] = warnings
    if pagination:
        meta["pagination"] = pagination
    meta.update(extra_meta)

    return SuccessResponse(
        data=data or {},
        meta=meta
    )
```

### `error_response`

Creates an error response envelope:

```python
@dataclass
class ErrorResponse:
    success: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=lambda: {"version": "response-v2"})

def error_response(
    error: str,
    data: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    **extra_meta
) -> ErrorResponse:
    """Create a standardized error response."""
    meta = {"version": "response-v2"}

    if request_id:
        meta["request_id"] = request_id
    meta.update(extra_meta)

    return ErrorResponse(
        error=error,
        data=data or {},
        meta=meta
    )
```

## Usage in Tools

### Basic Usage

```python
from dataclasses import asdict
from foundry_mcp.core.responses import success_response, error_response

@mcp.tool()
def get_user(user_id: str) -> dict:
    """Fetch a user by ID."""
    try:
        user = db.get_user(user_id)
        if not user:
            return asdict(error_response(
                error=f"User not found: {user_id}"
            ))

        return asdict(success_response(
            data={"user": user.to_dict()}
        ))
    except DatabaseError as e:
        return asdict(error_response(
            error=f"Database error: {str(e)}"
        ))
```

### With Warnings

```python
@mcp.tool()
def bulk_process(items: List[dict]) -> dict:
    """Process multiple items."""
    results = []
    warnings = []

    for item in items:
        try:
            results.append(process_item(item))
        except ItemError as e:
            warnings.append(f"Skipped item {item['id']}: {e}")

    return asdict(success_response(
        data={
            "processed": len(results),
            "results": results
        },
        warnings=warnings if warnings else None
    ))
```

### With Pagination

```python
@mcp.tool()
def list_items(cursor: Optional[str] = None, limit: int = 100) -> dict:
    """List items with pagination."""
    items, next_cursor, total = db.list_items(
        cursor=cursor,
        limit=limit
    )

    return asdict(success_response(
        data={"items": items},
        pagination={
            "cursor": next_cursor,
            "has_more": next_cursor is not None,
            "total_count": total,
            "page_size": limit
        }
    ))
```

## Specialized Helpers

### Paginated Response Helper

```python
def paginated_response(
    items: List[Any],
    cursor: Optional[str],
    has_more: bool,
    total_count: Optional[int] = None,
    item_key: str = "items"
) -> SuccessResponse:
    """Create a paginated response."""
    return success_response(
        data={item_key: items},
        pagination={
            "cursor": cursor,
            "has_more": has_more,
            "total_count": total_count
        }
    )
```

### Empty Success Helper

```python
def empty_success(message: str = "Operation completed") -> SuccessResponse:
    """Create an empty success response."""
    return success_response(
        data={"message": message}
    )
```

### Validation Error Helper

```python
def validation_error(
    errors: List[Dict[str, str]],
    message: str = "Validation failed"
) -> ErrorResponse:
    """Create a validation error response."""
    return error_response(
        error=f"{message}: {len(errors)} error(s)",
        data={"validation_errors": errors}
    )
```

## Telemetry Integration

Helpers should support telemetry without polluting business logic:

```python
import functools
import time

def with_telemetry(func):
    """Decorator to add telemetry to response."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000

        # Add telemetry to meta if result is a response dict
        if isinstance(result, dict) and "meta" in result:
            result["meta"]["telemetry"] = {
                "duration_ms": round(duration_ms, 2)
            }

        return result
    return wrapper

# Usage
@mcp.tool()
@with_telemetry
def expensive_operation() -> dict:
    result = do_work()
    return asdict(success_response(data=result))
```

## Anti-Patterns

### Don't: Manual Envelope Construction

```python
# Bad: Manual construction
@mcp.tool()
def bad_tool() -> dict:
    return {
        "success": True,
        "data": {"result": "..."},
        "error": None,
        "meta": {"version": "response-v2"}
    }

# Good: Use helpers
@mcp.tool()
def good_tool() -> dict:
    return asdict(success_response(
        data={"result": "..."}
    ))
```

### Don't: Mutate After Serialization

```python
# Bad: Mutating after asdict
@mcp.tool()
def bad_tool() -> dict:
    response = asdict(success_response(data={"result": "..."}))
    response["extra_field"] = "value"  # Bypasses schema!
    return response

# Good: Pass everything through helper
@mcp.tool()
def good_tool() -> dict:
    return asdict(success_response(
        data={"result": "...", "extra_field": "value"}
    ))
```

### Don't: Inconsistent Helper Usage

```python
# Bad: Mixed approaches in same codebase
def tool_a():
    return {"success": True, ...}  # Manual

def tool_b():
    return asdict(success_response(...))  # Helper

# Good: Consistent helper usage everywhere
def tool_a():
    return asdict(success_response(...))

def tool_b():
    return asdict(success_response(...))
```

## Testing Helpers

```python
def test_success_response_structure():
    """Verify success response has required fields."""
    response = asdict(success_response(data={"key": "value"}))

    assert response["success"] is True
    assert response["error"] is None
    assert response["data"] == {"key": "value"}
    assert response["meta"]["version"] == "response-v2"

def test_error_response_structure():
    """Verify error response has required fields."""
    response = asdict(error_response(error="Something failed"))

    assert response["success"] is False
    assert response["error"] == "Something failed"
    assert response["data"] == {}
    assert response["meta"]["version"] == "response-v2"
```

## Related Documents

- [Envelopes & Metadata](./02-envelopes-metadata.md) - Response structure
- [Testing & Fixtures](./10-testing-fixtures.md) - Testing patterns
- [Response Schema Reference](../codebase_standards/mcp_response_schema.md) - Full specification

---

**Navigation:** [← Envelopes & Metadata](./02-envelopes-metadata.md) | [Index](./README.md) | [Next: Validation & Input Hygiene →](./04-validation-input-hygiene.md)
