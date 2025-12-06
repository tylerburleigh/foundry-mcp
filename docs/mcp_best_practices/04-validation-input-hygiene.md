# 4. Validation, Typing & Input Hygiene

> Validate inputs early and reject malformed requests with descriptive errors.

## Overview

MCP tools receive inputs from various sources, including LLM-generated content that may be malformed or adversarial. Robust input validation is essential for security and reliability.

## Requirements

### MUST

- **Validate inputs early** - before any business logic executes
- **Reject invalid inputs** with descriptive error messages
- **Use typed schemas** (dataclasses, Pydantic, TypedDict) for input/output
- **Sanitize user data** before logging, storage, or downstream use

### SHOULD

- **Provide specific error messages** indicating which field failed and why
- **Validate in consistent order** (required fields, types, formats, business rules)
- **Use allowlists** over denylists for input validation

### MAY

- **Coerce compatible types** when unambiguous (e.g., "123" → 123)
- **Provide default values** for optional fields
- **Include validation hints** in tool descriptions

## Validation Order

Validate in this order for consistent, helpful error messages:

```
1. Required fields present
2. Type correctness
3. Format validation (regex, ranges)
4. Business rule validation
5. Cross-field validation
```

## Input Validation Patterns

### Using Pydantic (Recommended)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class CreateUserInput(BaseModel):
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    name: str = Field(..., min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    roles: List[str] = Field(default_factory=list)

    @validator('roles')
    def validate_roles(cls, v):
        allowed = {'admin', 'user', 'viewer'}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Invalid roles: {invalid}")
        return v

@mcp.tool()
def create_user(email: str, name: str, age: int = None, roles: list = None) -> dict:
    """Create a new user account."""
    try:
        validated = CreateUserInput(
            email=email,
            name=name,
            age=age,
            roles=roles or []
        )
    except ValidationError as e:
        return asdict(error_response(
            error="Validation failed",
            data={"validation_errors": e.errors()}
        ))

    # Proceed with validated data
    user = db.create_user(validated.dict())
    return asdict(success_response(data={"user": user}))
```

### Using Dataclasses with Manual Validation

```python
from dataclasses import dataclass
from typing import Optional, List
import re

@dataclass
class CreateUserInput:
    email: str
    name: str
    age: Optional[int] = None
    roles: List[str] = None

    def __post_init__(self):
        errors = []

        # Email validation
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', self.email):
            errors.append({"field": "email", "message": "Invalid email format"})

        # Name validation
        if not self.name or len(self.name) > 100:
            errors.append({"field": "name", "message": "Name must be 1-100 characters"})

        # Age validation
        if self.age is not None and not (0 <= self.age <= 150):
            errors.append({"field": "age", "message": "Age must be 0-150"})

        # Roles validation
        if self.roles:
            allowed = {'admin', 'user', 'viewer'}
            invalid = set(self.roles) - allowed
            if invalid:
                errors.append({"field": "roles", "message": f"Invalid roles: {invalid}"})

        if errors:
            raise ValueError(errors)
```

## Security-Critical Sanitization

### Prevent Injection Attacks

```python
import re
import html
from pathlib import Path

def sanitize_for_logging(value: str) -> str:
    """Remove control characters and limit length for safe logging."""
    # Remove control characters
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
    # Limit length
    return cleaned[:1000]

def sanitize_html(value: str) -> str:
    """Escape HTML entities to prevent XSS."""
    return html.escape(value)

def sanitize_path(value: str, base_dir: Path) -> Path:
    """Prevent path traversal attacks."""
    # Resolve to absolute path
    requested = (base_dir / value).resolve()

    # Ensure it's within base directory
    if not str(requested).startswith(str(base_dir.resolve())):
        raise ValueError("Path traversal detected")

    return requested

def sanitize_sql_identifier(value: str) -> str:
    """Validate SQL identifier (table/column names)."""
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', value):
        raise ValueError(f"Invalid SQL identifier: {value}")
    return value
```

### Prompt Injection Defense

MCP tools receiving LLM-generated input must be cautious:

```python
def validate_tool_input(input_text: str) -> str:
    """Validate input that may contain prompt injection attempts."""

    # Check for common injection patterns
    suspicious_patterns = [
        r'ignore\s+(previous|all)\s+instructions',
        r'system\s*:\s*',
        r'<\|.*?\|>',  # Special tokens
        r'\[INST\]',   # Instruction markers
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, input_text, re.IGNORECASE):
            raise ValueError("Input contains suspicious patterns")

    return input_text

@mcp.tool()
def process_text(text: str) -> dict:
    """Process user-provided text."""
    try:
        validated_text = validate_tool_input(text)
    except ValueError as e:
        return asdict(error_response(error=str(e)))

    # Safe to process
    result = do_processing(validated_text)
    return asdict(success_response(data={"result": result}))
```

## Size and Rate Limiting

### Recommended Limits

Define consistent limits across your MCP tools:

| Constant | Recommended Value | Purpose |
|----------|------------------|---------|
| `MAX_INPUT_SIZE` | 100,000 bytes (100KB) | Total request payload size |
| `MAX_ARRAY_LENGTH` | 1,000 items | Maximum array/list elements |
| `MAX_STRING_LENGTH` | 10,000 chars | Maximum string field length |
| `MAX_NESTED_DEPTH` | 10 levels | JSON nesting depth |
| `MAX_FIELD_COUNT` | 100 fields | Maximum object properties |

See [Security & Trust Boundaries § Size Limits](./08-security-trust-boundaries.md) for additional guidance on enforcing resource limits.

### Size Validation Helpers

```python
MAX_INPUT_SIZE = 100_000  # 100KB
MAX_ARRAY_LENGTH = 1000
MAX_STRING_LENGTH = 10_000

def validate_size_limits(data: dict) -> None:
    """Enforce size limits on input data."""
    import json

    serialized = json.dumps(data)
    if len(serialized) > MAX_INPUT_SIZE:
        raise ValueError(f"Input exceeds maximum size ({MAX_INPUT_SIZE} bytes)")

def validate_array_length(items: list, max_length: int = MAX_ARRAY_LENGTH) -> None:
    """Enforce array length limits."""
    if len(items) > max_length:
        raise ValueError(f"Array exceeds maximum length ({max_length} items)")

def validate_string_length(value: str, max_length: int = MAX_STRING_LENGTH) -> None:
    """Enforce string length limits."""
    if len(value) > max_length:
        raise ValueError(f"String exceeds maximum length ({max_length} characters)")
```

### Rate Limiting Integration

Use the built-in rate limiting infrastructure to enforce request limits:

```python
from foundry_mcp.core.rate_limit import (
    check_rate_limit,
    RateLimitConfig,
    get_rate_limit_manager
)
from foundry_mcp.core.responses import error_response

@mcp.tool()
def rate_limited_tool(user_id: str, data: dict) -> dict:
    """Tool with rate limiting enforcement."""
    # Check rate limit before processing
    result = check_rate_limit("rate_limited_tool", tenant_id=user_id)

    if not result.allowed:
        return asdict(error_response(
            error="Rate limit exceeded",
            data={
                "error_code": "RATE_LIMIT_EXCEEDED",
                "retry_after_seconds": result.reset_in,
                "limit": result.limit
            }
        ))

    # Proceed with validated, rate-limited request
    return asdict(success_response(data={"result": process(data)}))
```

Configure per-tool limits via environment variables:

```bash
# Global defaults
export FOUNDRY_RATE_LIMIT_DEFAULT=60  # requests per minute
export FOUNDRY_RATE_LIMIT_BURST=10    # burst limit

# Per-tool overrides
export FOUNDRY_RATE_LIMIT_RUN_TESTS=5
export FOUNDRY_RATE_LIMIT_EXPENSIVE_OPERATION=10
```

### Audit Logging for Validation Failures

Log validation failures for security monitoring:

```python
from foundry_mcp.core.observability import audit_log, get_audit_logger

def validate_with_audit(data: dict, user_id: str) -> bool:
    """Validate input and log failures for security review."""
    try:
        validated = InputSchema(**data)
        return True
    except ValidationError as e:
        # Log validation failure for security monitoring
        audit_log(
            "tool_invocation",
            tool="my_tool",
            success=False,
            user_id=user_id,
            error="validation_failed",
            validation_errors=str(e.errors())
        )
        return False

# For structured audit events
audit = get_audit_logger()

@mcp.tool()
def audited_tool(resource_id: str, action: str) -> dict:
    """Tool with comprehensive audit logging."""
    user_id = get_current_user()

    # Log access attempt
    audit.resource_access(
        resource_type="sensitive_data",
        resource_id=resource_id,
        action=action,
        user_id=user_id
    )

    # Validate and process...
    return result
```

See [Security & Trust Boundaries § Audit Logging](./08-security-trust-boundaries.md#audit-logging) for comprehensive audit logging patterns.

## Error Message Best Practices

### Good Error Messages

```python
# Specific and actionable
{
    "error": "Validation failed: 2 errors",
    "data": {
        "validation_errors": [
            {
                "field": "email",
                "message": "Invalid email format. Expected: user@domain.com",
                "received": "not-an-email"
            },
            {
                "field": "age",
                "message": "Must be a positive integer between 0 and 150",
                "received": -5
            }
        ]
    }
}
```

### Bad Error Messages

```python
# Too vague
{"error": "Invalid input"}

# Exposes internals
{"error": "SQLException: column 'email' violates constraint 'users_email_check'"}

# No guidance
{"error": "Validation error"}
```

## Anti-Patterns

### Don't: Validate After Processing

```python
# Bad: Process then validate
@mcp.tool()
def bad_tool(data: dict) -> dict:
    result = expensive_operation(data)  # Wasted work if invalid
    if not is_valid(data):
        return error_response(...)

# Good: Validate first
@mcp.tool()
def good_tool(data: dict) -> dict:
    if not is_valid(data):
        return error_response(...)
    result = expensive_operation(data)
```

### Don't: Trust Input Types

```python
# Bad: Assume correct type
@mcp.tool()
def bad_tool(count: int) -> dict:
    for i in range(count):  # Crashes if count is string
        ...

# Good: Validate type
@mcp.tool()
def good_tool(count: int) -> dict:
    if not isinstance(count, int) or count < 0:
        return error_response("count must be a non-negative integer")
    for i in range(count):
        ...
```

### Don't: Use Denylists for Security

```python
# Bad: Denylist approach
dangerous_chars = ['<', '>', '&', '"', "'"]
if any(c in user_input for c in dangerous_chars):
    raise ValueError("Dangerous characters detected")

# Good: Allowlist approach
allowed_pattern = r'^[a-zA-Z0-9\s\-_\.]+$'
if not re.match(allowed_pattern, user_input):
    raise ValueError("Input contains invalid characters")
```

## Related Documents

- [Security & Trust Boundaries](./08-security-trust-boundaries.md) - Comprehensive security patterns including:
  - [Prompt Injection Defense](./08-security-trust-boundaries.md#prompt-injection-defense)
  - [Authorization Patterns](./08-security-trust-boundaries.md#authorization-patterns)
  - [Rate Limiting](./08-security-trust-boundaries.md#rate-limiting)
  - [Audit Logging](./08-security-trust-boundaries.md#audit-logging)
- [Error Semantics](./07-error-semantics.md) - Error handling and response patterns
- [AI/LLM Integration](./11-ai-llm-integration.md) - LLM-specific security concerns
- [Observability & Telemetry](./05-observability-telemetry.md) - Metrics and structured logging

---

**Navigation:** [← Serialization Helpers](./03-serialization-helpers.md) | [Index](./README.md) | [Next: Observability & Telemetry →](./05-observability-telemetry.md)
