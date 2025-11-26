# 12. Timeout & Resilience Patterns

> Handle failures gracefully with timeouts, retries, and circuit breakers.

## Overview

MCP tools interact with external services, databases, and APIs that can fail or become slow. Resilience patterns ensure tools remain responsive and recover gracefully from failures.

## Requirements

### MUST

- **Set explicit timeouts** - never wait indefinitely
- **Handle timeout errors gracefully** - return meaningful errors
- **Clean up resources on failure** - prevent leaks
- **Document timeout behavior** - in tool descriptions

### SHOULD

- **Implement circuit breakers** - for external dependencies
- **Use exponential backoff** - for retries
- **Provide partial results** - when possible on timeout
- **Track reliability metrics** - failure rates, latencies

### MAY

- **Support timeout configuration** - per-request overrides
- **Implement bulkhead isolation** - prevent cascade failures
- **Use fallback strategies** - degraded but functional responses

## Timeout Budgets

### Timeout Categories

| Category | Default | Max | Use Case |
|----------|---------|-----|----------|
| Fast | 5s | 10s | Cache lookups, simple queries |
| Medium | 30s | 60s | Database operations, API calls |
| Slow | 120s | 300s | File processing, complex operations |
| Background | 600s | 3600s | Batch jobs, large transfers |

### Implementing Timeouts

```python
import asyncio
from functools import wraps
from typing import TypeVar, Callable, Any

T = TypeVar('T')

class TimeoutError(Exception):
    """Operation timed out."""
    pass

def with_timeout(seconds: float, error_message: str = None):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                msg = error_message or f"{func.__name__} timed out after {seconds}s"
                raise TimeoutError(msg)
        return wrapper
    return decorator

# Usage
@mcp.tool()
@with_timeout(30, "Database query timed out")
async def query_database(query: str) -> dict:
    """Execute database query with 30s timeout."""
    result = await db.execute(query)
    return asdict(success_response(data={"result": result}))
```

### Synchronous Timeout

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds: int):
    """Context manager for synchronous timeout."""
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    original = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original)

# Usage
@mcp.tool()
def process_file(file_path: str) -> dict:
    """Process file with timeout."""
    try:
        with timeout(60):
            result = heavy_processing(file_path)
        return asdict(success_response(data=result))
    except TimeoutError:
        return asdict(error_response(
            error="File processing timed out after 60 seconds",
            data={"error_code": "TIMEOUT", "timeout_seconds": 60}
        ))
```

## Retry Patterns

### Exponential Backoff

```python
import random
import time
from typing import Callable, TypeVar, List, Type

T = TypeVar('T')

def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: List[Type[Exception]] = None
) -> T:
    """Retry function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Multiplier for each retry
        jitter: Add randomness to prevent thundering herd
        retryable_exceptions: Exceptions that trigger retry
    """
    retryable = tuple(retryable_exceptions or [Exception])
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable as e:
            last_exception = e

            if attempt == max_retries:
                break

            # Calculate delay
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            if jitter:
                delay = delay * (0.5 + random.random())

            time.sleep(delay)

    raise last_exception

# Usage
@mcp.tool()
def fetch_external_data(url: str) -> dict:
    """Fetch data from external API with retries."""
    try:
        result = retry_with_backoff(
            lambda: http_client.get(url),
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions=[ConnectionError, TimeoutError]
        )
        return asdict(success_response(data={"response": result}))
    except Exception as e:
        return asdict(error_response(
            error=f"Failed after 3 retries: {str(e)}",
            data={"error_code": "EXTERNAL_SERVICE_FAILURE"}
        ))
```

### Retry Decorator

```python
from functools import wraps
from typing import Tuple

def retryable(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator for automatic retries."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=delay,
                retryable_exceptions=list(exceptions)
            )
        return wrapper
    return decorator

# Usage
@retryable(max_retries=3, exceptions=(ConnectionError,))
def call_api(endpoint: str):
    return http_client.get(endpoint)
```

## Circuit Breaker

```python
import time
from enum import Enum
from dataclasses import dataclass
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreaker:
    """Circuit breaker for external dependencies."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    def __post_init__(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self.lock = Lock()

    def can_execute(self) -> bool:
        """Check if request should proceed."""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery timeout elapsed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False

        return False

    def record_success(self):
        """Record successful call."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    # Recovery successful
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            else:
                self.failure_count = 0

    def record_failure(self):
        """Record failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # Recovery failed, back to open
                self.state = CircuitState.OPEN

            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

# Global circuit breakers per dependency
circuit_breakers = {
    "database": CircuitBreaker(failure_threshold=5, recovery_timeout=30),
    "external_api": CircuitBreaker(failure_threshold=3, recovery_timeout=60),
}

def with_circuit_breaker(breaker_name: str):
    """Decorator to wrap function with circuit breaker."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            breaker = circuit_breakers[breaker_name]

            if not breaker.can_execute():
                raise Exception(f"Circuit breaker '{breaker_name}' is open")

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise

        return wrapper
    return decorator

# Usage
@mcp.tool()
def query_database(query: str) -> dict:
    """Query database with circuit breaker protection."""
    try:
        result = _execute_query(query)
        return asdict(success_response(data={"result": result}))
    except Exception as e:
        if "Circuit breaker" in str(e):
            return asdict(error_response(
                error="Database temporarily unavailable",
                data={
                    "error_code": "SERVICE_UNAVAILABLE",
                    "retry_after_seconds": 30
                }
            ))
        return asdict(error_response(f"Query failed: {str(e)}"))

@with_circuit_breaker("database")
def _execute_query(query: str):
    return db.execute(query)
```

## Partial Results on Timeout

```python
import asyncio
from typing import List

@mcp.tool()
async def batch_fetch(urls: List[str], timeout_per_url: float = 5.0) -> dict:
    """Fetch multiple URLs, returning partial results on timeout.

    Individual URL timeouts don't fail the entire batch.
    """
    results = []
    errors = []

    async def fetch_one(url: str) -> dict:
        try:
            async with asyncio.timeout(timeout_per_url):
                response = await http_client.get(url)
                return {"url": url, "status": "success", "data": response}
        except asyncio.TimeoutError:
            return {"url": url, "status": "timeout"}
        except Exception as e:
            return {"url": url, "status": "error", "error": str(e)}

    # Fetch all concurrently
    fetch_results = await asyncio.gather(
        *[fetch_one(url) for url in urls],
        return_exceptions=False
    )

    for result in fetch_results:
        if result["status"] == "success":
            results.append(result)
        else:
            errors.append(result)

    warnings = []
    if errors:
        timeout_count = sum(1 for e in errors if e["status"] == "timeout")
        error_count = len(errors) - timeout_count
        if timeout_count:
            warnings.append(f"{timeout_count} URLs timed out")
        if error_count:
            warnings.append(f"{error_count} URLs failed")

    return asdict(success_response(
        data={
            "results": results,
            "errors": errors,
            "total_requested": len(urls),
            "total_succeeded": len(results)
        },
        warnings=warnings if warnings else None
    ))
```

## Resource Cleanup

```python
from contextlib import asynccontextmanager
from typing import AsyncIterator

@asynccontextmanager
async def managed_connection(timeout: float = 30.0) -> AsyncIterator:
    """Context manager ensuring connection cleanup."""
    conn = None
    try:
        conn = await asyncio.wait_for(
            db.connect(),
            timeout=timeout
        )
        yield conn
    except asyncio.TimeoutError:
        raise TimeoutError("Connection timed out")
    finally:
        if conn:
            try:
                await asyncio.wait_for(conn.close(), timeout=5.0)
            except:
                pass  # Best effort cleanup

@mcp.tool()
async def safe_query(query: str) -> dict:
    """Query with guaranteed resource cleanup."""
    try:
        async with managed_connection(timeout=10.0) as conn:
            result = await asyncio.wait_for(
                conn.execute(query),
                timeout=30.0
            )
            return asdict(success_response(data={"result": result}))
    except TimeoutError as e:
        return asdict(error_response(str(e)))
```

## Health Checks

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

@dataclass
class HealthStatus:
    healthy: bool
    latency_ms: float
    last_check: datetime
    error: str = None

health_cache: Dict[str, HealthStatus] = {}

async def check_dependency_health(name: str, check_func) -> HealthStatus:
    """Check health of a dependency."""
    start = time.perf_counter()
    try:
        await asyncio.wait_for(check_func(), timeout=5.0)
        latency = (time.perf_counter() - start) * 1000
        status = HealthStatus(
            healthy=True,
            latency_ms=latency,
            last_check=datetime.utcnow()
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        status = HealthStatus(
            healthy=False,
            latency_ms=latency,
            last_check=datetime.utcnow(),
            error=str(e)
        )

    health_cache[name] = status
    return status

@mcp.tool()
async def health_check() -> dict:
    """Check health of all dependencies."""
    checks = {
        "database": lambda: db.execute("SELECT 1"),
        "cache": lambda: cache.ping(),
        "external_api": lambda: http_client.get(API_HEALTH_URL),
    }

    results = {}
    for name, check_func in checks.items():
        status = await check_dependency_health(name, check_func)
        results[name] = {
            "healthy": status.healthy,
            "latency_ms": status.latency_ms,
            "error": status.error
        }

    all_healthy = all(r["healthy"] for r in results.values())

    return asdict(success_response(
        data={
            "status": "healthy" if all_healthy else "degraded",
            "dependencies": results
        }
    ))
```

## Anti-Patterns

### Don't: Wait Indefinitely

```python
# Bad: No timeout
result = external_api.fetch(url)

# Good: Explicit timeout
result = await asyncio.wait_for(external_api.fetch(url), timeout=30)
```

### Don't: Retry Without Backoff

```python
# Bad: Immediate retries (hammers failing service)
for _ in range(3):
    try:
        return api.call()
    except:
        pass

# Good: Exponential backoff
retry_with_backoff(api.call, base_delay=1.0, exponential_base=2.0)
```

### Don't: Leak Resources on Failure

```python
# Bad: Connection leak on timeout
conn = db.connect()
result = conn.execute(query)  # If this times out, conn is leaked
conn.close()

# Good: Context manager ensures cleanup
async with managed_connection() as conn:
    result = await conn.execute(query)
```

## Related Documents

- [Error Semantics](./07-error-semantics.md) - Error handling
- [Observability & Telemetry](./05-observability-telemetry.md) - Tracking failures
- [Pagination & Streaming](./06-pagination-streaming.md) - Handling large operations

---

**Navigation:** [← AI/LLM Integration](./11-ai-llm-integration.md) | [Index](./README.md) | [Next: Tool Discovery →](./13-tool-discovery.md)
