# 5. Observability & Telemetry

> Emit structured logs and metrics for debugging, monitoring, and SLO tracking.

## Overview

Observability enables understanding system behavior in production. MCP tools should emit structured telemetry that supports debugging, performance monitoring, and SLO measurement.

## Requirements

### MUST

- **Use structured logging** (JSON format recommended)
- **Include correlation IDs** in all log entries
- **Log operation outcomes** (success/failure, duration)
- **Redact sensitive data** from logs and telemetry

### SHOULD

- **Integrate distributed tracing** (OpenTelemetry recommended)
- **Track key metrics** (latency, error rates, throughput)
- **Use consistent log levels** across all tools
- **Include context** (operation name, user ID, resource ID)

### MAY

- **Expose Prometheus metrics** endpoints
- **Include detailed timing breakdowns** in telemetry
- **Track business metrics** alongside technical metrics

## Structured Logging

### Log Format (JSON)

```json
{
    "timestamp": "2025-11-26T10:30:00.123Z",
    "level": "INFO",
    "message": "Operation completed",
    "operation": "get_user",
    "request_id": "req_abc123",
    "trace_id": "trace_xyz789",
    "span_id": "span_456",
    "user_id": "usr_123",
    "duration_ms": 45,
    "success": true,
    "metadata": {
        "cache_hit": true,
        "db_queries": 1
    }
}
```

### Log Levels

| Level | Use For | Example |
|-------|---------|---------|
| `ERROR` | Failures requiring attention | Database connection lost |
| `WARN` | Degraded but functional | Cache miss, retry succeeded |
| `INFO` | Normal operations | Request completed |
| `DEBUG` | Detailed troubleshooting | Query parameters, intermediate values |

### Python Logging Setup

```python
import logging
import json
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "operation"):
            log_data["operation"] = record.operation
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "metadata"):
            log_data["metadata"] = record.metadata

        return json.dumps(log_data)

# Setup
logger = logging.getLogger("foundry_mcp")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Logging Decorator

```python
import functools
import time
import uuid
from contextvars import ContextVar

# Context variable for request ID propagation
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

def with_logging(operation_name: str):
    """Decorator to add structured logging to tool functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            request_id = request_id_var.get() or str(uuid.uuid4())[:8]
            start_time = time.perf_counter()

            logger.info(
                f"Starting {operation_name}",
                extra={
                    "operation": operation_name,
                    "request_id": request_id
                }
            )

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                success = result.get("success", True) if isinstance(result, dict) else True

                logger.info(
                    f"Completed {operation_name}",
                    extra={
                        "operation": operation_name,
                        "request_id": request_id,
                        "duration_ms": round(duration_ms, 2),
                        "success": success
                    }
                )

                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Failed {operation_name}: {str(e)}",
                    extra={
                        "operation": operation_name,
                        "request_id": request_id,
                        "duration_ms": round(duration_ms, 2),
                        "error": str(e)
                    }
                )
                raise

        return wrapper
    return decorator

# Usage
@mcp.tool()
@with_logging("get_user")
def get_user(user_id: str) -> dict:
    ...
```

## Distributed Tracing (OpenTelemetry)

### Setup

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer("foundry_mcp")

# Configure exporter
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)
```

### Tracing Decorator

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("foundry_mcp")

def with_tracing(operation_name: str):
    """Decorator to add distributed tracing to tool functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(operation_name) as span:
                # Add attributes
                span.set_attribute("mcp.tool", operation_name)

                try:
                    result = func(*args, **kwargs)

                    if isinstance(result, dict):
                        success = result.get("success", True)
                        if not success:
                            span.set_status(Status(StatusCode.ERROR))
                            span.set_attribute("error.message", result.get("error", ""))

                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator

# Usage
@mcp.tool()
@with_tracing("create_project")
def create_project(name: str) -> dict:
    with tracer.start_as_current_span("validate_input"):
        validated = validate_input(name)

    with tracer.start_as_current_span("db_insert"):
        project = db.create_project(validated)

    return asdict(success_response(data={"project": project}))
```

## Metrics

### Key Metrics to Track

| Metric | Type | Description |
|--------|------|-------------|
| `mcp_tool_requests_total` | Counter | Total requests per tool |
| `mcp_tool_duration_seconds` | Histogram | Request latency |
| `mcp_tool_errors_total` | Counter | Error count by type |
| `mcp_tool_in_progress` | Gauge | Currently executing requests |

### Prometheus Integration

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUESTS = Counter(
    'mcp_tool_requests_total',
    'Total tool requests',
    ['tool', 'status']
)

DURATION = Histogram(
    'mcp_tool_duration_seconds',
    'Tool request duration',
    ['tool'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10]
)

IN_PROGRESS = Gauge(
    'mcp_tool_in_progress',
    'Currently executing requests',
    ['tool']
)

def with_metrics(tool_name: str):
    """Decorator to record metrics for tool functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            IN_PROGRESS.labels(tool=tool_name).inc()

            with DURATION.labels(tool=tool_name).time():
                try:
                    result = func(*args, **kwargs)

                    status = "success" if result.get("success", True) else "error"
                    REQUESTS.labels(tool=tool_name, status=status).inc()

                    return result

                except Exception:
                    REQUESTS.labels(tool=tool_name, status="exception").inc()
                    raise

                finally:
                    IN_PROGRESS.labels(tool=tool_name).dec()

        return wrapper
    return decorator
```

## Sensitive Data Redaction

```python
import re
from typing import Any

REDACT_PATTERNS = [
    (r'password["\']?\s*[:=]\s*["\']?[^"\'&\s]+', 'password=***REDACTED***'),
    (r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w\-]+', 'api_key=***REDACTED***'),
    (r'bearer\s+[\w\-\.]+', 'Bearer ***REDACTED***'),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***EMAIL***'),
]

def redact_sensitive(value: Any) -> Any:
    """Redact sensitive data from log values."""
    if isinstance(value, str):
        result = value
        for pattern, replacement in REDACT_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    elif isinstance(value, dict):
        return {k: redact_sensitive(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value

# Usage in logging
logger.info("Request received", extra={
    "request_data": redact_sensitive(request_data)
})
```

## Anti-Patterns

### Don't: Use Print Statements

```python
# Bad: Print statements
print(f"Processing user {user_id}")
print(f"Error: {e}")

# Good: Structured logging
logger.info("Processing user", extra={"user_id": user_id})
logger.error("Operation failed", extra={"error": str(e)})
```

### Don't: Log Sensitive Data

```python
# Bad: Logging credentials
logger.info(f"Authenticating with API key: {api_key}")

# Good: Redact sensitive data
logger.info("Authenticating", extra={"api_key_prefix": api_key[:4] + "***"})
```

### Don't: Inconsistent Log Levels

```python
# Bad: INFO for errors
logger.info(f"Failed to connect: {error}")

# Good: Appropriate level
logger.error("Failed to connect", extra={"error": str(error)})
```

## Related Documents

- [Resilience Patterns](./12-timeout-resilience.md) - Timeout and retry tracking
- [Error Semantics](./07-error-semantics.md) - Error classification
- [Security & Trust Boundaries](./08-security-trust-boundaries.md) - Data redaction

---

**Navigation:** [← Validation & Input Hygiene](./04-validation-input-hygiene.md) | [Index](./README.md) | [Next: Pagination & Streaming →](./06-pagination-streaming.md)
