# 8. Security & Trust Boundaries

> Treat all inputs as untrusted and enforce security at every boundary.

## Overview

MCP tools operate at trust boundaries between LLMs, users, and system resources. Security must be enforced proactively, not assumed. This is especially critical for AI-integrated tools where inputs may be LLM-generated.

## Requirements

### MUST

- **Treat all inputs as untrusted** - including LLM-generated content
- **Validate and sanitize inputs** - before any processing
- **Enforce authorization** - check permissions before privileged operations
- **Redact secrets from logs/responses** - credentials, PII, tokens
- **Limit resource access** - files, network, memory

### SHOULD

- **Implement rate limiting** - per-user and per-operation
- **Audit security-sensitive operations** - who did what, when
- **Use allowlists over denylists** - for input validation
- **Apply principle of least privilege** - minimal required permissions

### MAY

- **Sandbox untrusted operations** - isolated execution environments
- **Implement capability-based security** - fine-grained permissions
- **Support security headers** - CSP, CORS for web contexts

## Trust Boundaries in MCP

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM Context                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              MCP Tool Input (UNTRUSTED)              │   │
│  │  - May contain prompt injection                      │   │
│  │  - May be malformed or adversarial                   │   │
│  │  - May exceed expected sizes                         │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Validation & Sanitization Layer           │   │
│  │  - Input validation                                 │   │
│  │  - Size limits                                      │   │
│  │  - Format verification                              │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Authorization Layer                     │   │
│  │  - Permission checks                                │   │
│  │  - Rate limiting                                    │   │
│  │  - Capability verification                          │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Business Logic (TRUSTED)               │   │
│  │  - Only executed after validation passes            │   │
│  │  - Operates on sanitized data                       │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         External Resources (PROTECTED)              │   │
│  │  - Database, filesystem, network                    │   │
│  │  - Accessed with minimal privileges                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Prompt Injection Defense

MCP tools receiving LLM-generated input are vulnerable to prompt injection:

```python
import re
from typing import Tuple

INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)',
    r'disregard\s+(all\s+)?(previous|prior|above)',
    r'forget\s+(everything|all)\s+(above|before)',
    r'new\s+instructions?\s*:',
    r'system\s*:\s*',
    r'<\|.*?\|>',           # Special tokens
    r'\[INST\]|\[/INST\]',  # Instruction markers
    r'```system',           # Code block injection
]

def detect_injection(text: str) -> Tuple[bool, str]:
    """Detect potential prompt injection attempts.

    Returns:
        Tuple of (is_suspicious, matched_pattern)
    """
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True, pattern
    return False, ""

@mcp.tool()
def process_user_text(text: str) -> dict:
    """Process user-provided text with injection protection."""
    # Check for injection attempts
    is_suspicious, pattern = detect_injection(text)
    if is_suspicious:
        # Log for security monitoring
        logger.warning("Potential injection detected", extra={
            "pattern": pattern,
            "text_preview": text[:100]
        })
        return asdict(error_response(
            error="Input contains disallowed patterns",
            data={"error_code": "SUSPICIOUS_INPUT"}
        ))

    # Safe to proceed
    result = do_processing(text)
    return asdict(success_response(data=result))
```

## Input Sanitization

### Path Traversal Prevention

```python
from pathlib import Path

def safe_path(user_path: str, base_dir: Path) -> Path:
    """Resolve path safely within base directory.

    Raises:
        ValueError: If path escapes base directory
    """
    # Resolve to absolute path
    base = base_dir.resolve()
    requested = (base / user_path).resolve()

    # Verify containment
    if not str(requested).startswith(str(base) + "/"):
        raise ValueError(f"Path '{user_path}' escapes base directory")

    return requested

@mcp.tool()
def read_file(path: str) -> dict:
    """Read a file from the allowed directory."""
    try:
        safe = safe_path(path, ALLOWED_DIR)
        content = safe.read_text()
        return asdict(success_response(data={"content": content}))
    except ValueError as e:
        return asdict(error_response(f"Access denied: {e}"))
    except FileNotFoundError:
        return asdict(error_response(f"File not found: {path}"))
```

### SQL Injection Prevention

```python
# Bad: String interpolation
def bad_query(user_id: str):
    cursor.execute(f"SELECT * FROM users WHERE id = '{user_id}'")

# Good: Parameterized queries
def good_query(user_id: str):
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# Good: ORM with automatic escaping
def orm_query(user_id: str):
    return User.query.filter_by(id=user_id).first()
```

### Command Injection Prevention

```python
import shlex
import subprocess

# Bad: Shell injection vulnerability
def bad_command(filename: str):
    subprocess.run(f"cat {filename}", shell=True)

# Good: Avoid shell, use list
def good_command(filename: str):
    # Validate filename first
    if not re.match(r'^[\w\-\.]+$', filename):
        raise ValueError("Invalid filename")
    subprocess.run(["cat", filename], shell=False)

# Better: Don't use subprocess for file reading
def better_approach(filename: str):
    safe = safe_path(filename, ALLOWED_DIR)
    return safe.read_text()
```

## Authorization Patterns

### Permission Checking

```python
from enum import Enum
from typing import Set

class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

def check_permission(
    user_id: str,
    resource_id: str,
    required: Permission
) -> bool:
    """Check if user has permission on resource."""
    user_permissions = get_user_permissions(user_id, resource_id)
    return required in user_permissions or Permission.ADMIN in user_permissions

def require_permission(required: Permission):
    """Decorator to enforce permission check."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_id = get_current_user_id()
            resource_id = kwargs.get("resource_id") or args[0]

            if not check_permission(user_id, resource_id, required):
                return asdict(error_response(
                    error=f"Permission denied: requires '{required}' on '{resource_id}'",
                    data={
                        "error_code": "PERMISSION_DENIED",
                        "required_permission": required
                    }
                ))

            return func(*args, **kwargs)
        return wrapper
    return decorator

@mcp.tool()
@require_permission(Permission.DELETE)
def delete_resource(resource_id: str) -> dict:
    """Delete a resource (requires delete permission)."""
    db.delete(resource_id)
    return asdict(success_response(data={"deleted": resource_id}))
```

### Rate Limiting

```python
from datetime import datetime, timedelta
from collections import defaultdict
import threading

class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = timedelta(seconds=window_seconds)
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """Check if request is allowed.

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = datetime.utcnow()
        cutoff = now - self.window

        with self.lock:
            # Clean old requests
            self.requests[key] = [
                ts for ts in self.requests[key] if ts > cutoff
            ]

            if len(self.requests[key]) >= self.limit:
                # Calculate retry time
                oldest = min(self.requests[key])
                retry_after = int((oldest + self.window - now).total_seconds()) + 1
                return False, retry_after

            self.requests[key].append(now)
            return True, 0

# Usage
limiter = RateLimiter(limit=100, window_seconds=60)

@mcp.tool()
def rate_limited_operation(user_id: str) -> dict:
    """Operation with rate limiting."""
    allowed, retry_after = limiter.is_allowed(user_id)

    if not allowed:
        return asdict(error_response(
            error="Rate limit exceeded",
            data={
                "error_code": "RATE_LIMIT_EXCEEDED",
                "retry_after_seconds": retry_after
            }
        ))

    # Proceed with operation
    return asdict(success_response(data={"result": "..."}))
```

## Secret Redaction

```python
import re
from typing import Any

# Patterns for sensitive data
SECRET_PATTERNS = {
    "api_key": r'(?i)(api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?[\w\-]+',
    "password": r'(?i)(password|passwd|pwd)["\']?\s*[:=]\s*["\']?[^\s"\']+',
    "token": r'(?i)(token|bearer|auth)["\']?\s*[:=]\s*["\']?[\w\-\.]+',
    "secret": r'(?i)(secret|private[_-]?key)["\']?\s*[:=]\s*["\']?[\w\-]+',
}

def redact_secrets(data: Any) -> Any:
    """Recursively redact secrets from data structure."""
    if isinstance(data, str):
        result = data
        for name, pattern in SECRET_PATTERNS.items():
            result = re.sub(pattern, f'{name}=***REDACTED***', result)
        return result
    elif isinstance(data, dict):
        return {k: redact_secrets(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_secrets(item) for item in data]
    return data

# Use in logging and responses
logger.info("Request", extra={"data": redact_secrets(request_data)})
```

## Audit Logging

```python
from datetime import datetime
import json

def audit_log(
    action: str,
    user_id: str,
    resource_type: str,
    resource_id: str,
    outcome: str,
    details: dict = None
):
    """Log security-sensitive action for audit trail."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "action": action,
        "user_id": user_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "outcome": outcome,  # "success", "denied", "error"
        "details": details or {}
    }

    # Write to audit log (separate from application logs)
    audit_logger.info(json.dumps(entry))

# Usage
@mcp.tool()
def delete_project(project_id: str) -> dict:
    """Delete a project with audit logging."""
    user_id = get_current_user_id()

    if not check_permission(user_id, project_id, Permission.DELETE):
        audit_log(
            action="delete_project",
            user_id=user_id,
            resource_type="project",
            resource_id=project_id,
            outcome="denied",
            details={"reason": "insufficient_permissions"}
        )
        return asdict(error_response("Permission denied"))

    db.delete_project(project_id)

    audit_log(
        action="delete_project",
        user_id=user_id,
        resource_type="project",
        resource_id=project_id,
        outcome="success"
    )

    return asdict(success_response(data={"deleted": project_id}))
```

## Multi-Tenancy & Isolation

When serving multiple clients or tenants, ensure proper isolation:

### Tenant Context

```python
from contextvars import ContextVar
from typing import Optional

# Tenant context for current request
current_tenant: ContextVar[str] = ContextVar("tenant_id", default="")

def get_tenant_id() -> str:
    """Get current tenant ID or raise if not set."""
    tenant = current_tenant.get()
    if not tenant:
        raise SecurityError("Tenant context not established")
    return tenant

def require_tenant(func):
    """Decorator to require tenant context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not current_tenant.get():
            return asdict(error_response(
                error="Tenant context required",
                data={"error_code": "TENANT_REQUIRED"}
            ))
        return func(*args, **kwargs)
    return wrapper
```

### Data Isolation

```python
class TenantAwareQuery:
    """Ensure all queries are scoped to current tenant."""

    def __init__(self, db_session):
        self.session = db_session

    def query(self, model, **filters):
        """Query with automatic tenant filtering."""
        tenant_id = get_tenant_id()

        # Always include tenant filter
        return self.session.query(model).filter(
            model.tenant_id == tenant_id,
            **filters
        )

    def create(self, model, **data):
        """Create with automatic tenant assignment."""
        tenant_id = get_tenant_id()
        data["tenant_id"] = tenant_id
        return self.session.create(model, **data)

# Usage
@mcp.tool()
@require_tenant
def list_projects() -> dict:
    """List projects for current tenant."""
    # tenant_id automatically applied
    projects = tenant_db.query(Project).all()
    return asdict(success_response(data={"projects": projects}))
```

### Tenant-Scoped Rate Limiting

```python
from collections import defaultdict
from datetime import datetime, timedelta

class TenantRateLimiter:
    """Rate limiting per tenant."""

    def __init__(self, default_limit: int = 100, window_seconds: int = 60):
        self.default_limit = default_limit
        self.window = timedelta(seconds=window_seconds)
        self.requests = defaultdict(list)
        self.custom_limits = {}  # tenant_id -> limit

    def set_tenant_limit(self, tenant_id: str, limit: int):
        """Set custom limit for specific tenant."""
        self.custom_limits[tenant_id] = limit

    def is_allowed(self, tenant_id: str) -> tuple[bool, int]:
        """Check if request is allowed for tenant.

        Returns: (is_allowed, retry_after_seconds)
        """
        now = datetime.utcnow()
        limit = self.custom_limits.get(tenant_id, self.default_limit)

        # Clean old requests
        cutoff = now - self.window
        self.requests[tenant_id] = [
            ts for ts in self.requests[tenant_id] if ts > cutoff
        ]

        if len(self.requests[tenant_id]) >= limit:
            oldest = min(self.requests[tenant_id])
            retry_after = int((oldest + self.window - now).total_seconds()) + 1
            return False, retry_after

        self.requests[tenant_id].append(now)
        return True, 0

tenant_limiter = TenantRateLimiter(default_limit=100, window_seconds=60)

@mcp.tool()
def tenant_rate_limited_operation() -> dict:
    """Operation with per-tenant rate limiting."""
    tenant_id = get_tenant_id()
    allowed, retry_after = tenant_limiter.is_allowed(tenant_id)

    if not allowed:
        return asdict(error_response(
            error="Rate limit exceeded for tenant",
            data={
                "error_code": "RATE_LIMIT_EXCEEDED",
                "tenant_id": tenant_id,
                "retry_after_seconds": retry_after
            }
        ))

    return asdict(success_response(data={"result": "..."}))
```

### Tenant ID in Logs (Redacted)

```python
def log_with_tenant(message: str, **extra):
    """Log with tenant context, redacting if needed."""
    tenant_id = current_tenant.get()

    # Option 1: Include full tenant ID (internal logs)
    extra["tenant_id"] = tenant_id

    # Option 2: Redact tenant ID (external/shared logs)
    # extra["tenant_id_prefix"] = tenant_id[:8] + "..."

    logger.info(message, extra=extra)
```

### Cross-Tenant Access Prevention

```python
def validate_resource_tenant(resource_id: str, resource_type: str) -> bool:
    """Verify resource belongs to current tenant."""
    tenant_id = get_tenant_id()
    resource = db.get(resource_type, resource_id)

    if not resource:
        return False  # Not found

    if resource.tenant_id != tenant_id:
        # Log potential security issue
        logger.warning(
            "Cross-tenant access attempt",
            extra={
                "tenant_id": tenant_id,
                "resource_tenant": resource.tenant_id,
                "resource_type": resource_type,
                "resource_id": resource_id
            }
        )
        return False

    return True

@mcp.tool()
@require_tenant
def get_resource(resource_id: str) -> dict:
    """Get resource with tenant validation."""
    if not validate_resource_tenant(resource_id, "resource"):
        return asdict(error_response(
            error="Resource not found",  # Don't reveal it exists for other tenant
            data={"error_code": "NOT_FOUND"}
        ))

    resource = db.get("resource", resource_id)
    return asdict(success_response(data={"resource": resource}))
```

## Anti-Patterns

### Don't: Trust Input Sources

```python
# Bad: Assuming LLM input is safe
@mcp.tool()
def bad_tool(query: str):
    # Directly using LLM-provided SQL!
    db.execute(query)

# Good: Never execute arbitrary input
@mcp.tool()
def good_tool(table: str, filters: dict):
    # Validate table name against allowlist
    if table not in ALLOWED_TABLES:
        return error_response("Invalid table")
    # Use parameterized queries
    return db.query(table, filters)
```

### Don't: Log Secrets

```python
# Bad: Logging credentials
logger.info(f"Connecting with password: {password}")

# Good: Redact sensitive data
logger.info(f"Connecting as user: {username}")
```

### Don't: Return Internal Errors

```python
# Bad: Exposing internals
return {"error": str(exception)}  # May contain paths, SQL, etc.

# Good: Generic message, log details
logger.error("Operation failed", exc_info=True)
return {"error": "An internal error occurred", "request_id": request_id}
```

## Related Documents

- [Validation & Input Hygiene](./04-validation-input-hygiene.md) - Input validation
- [AI/LLM Integration](./11-ai-llm-integration.md) - LLM-specific security
- [Observability & Telemetry](./05-observability-telemetry.md) - Audit logging

---

**Navigation:** [← Error Semantics](./07-error-semantics.md) | [Index](./README.md) | [Next: Spec-Driven Development →](./09-spec-driven-development.md)
