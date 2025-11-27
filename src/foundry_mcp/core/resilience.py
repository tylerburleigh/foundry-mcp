"""
Resilience primitives for MCP tool operations.

Provides timeout budgets, retry patterns, circuit breakers, and health checks
for building robust MCP tools that handle failures gracefully.

Timeout Budget Categories
=========================

Use the appropriate timeout category based on operation type:

    FAST_TIMEOUT (5s)       - Cache lookups, simple queries
    MEDIUM_TIMEOUT (30s)    - Database operations, API calls
    SLOW_TIMEOUT (120s)     - File processing, complex operations
    BACKGROUND_TIMEOUT (600s) - Batch jobs, large transfers

Example usage:

    from foundry_mcp.core.resilience import (
        MEDIUM_TIMEOUT,
        with_timeout,
        retry_with_backoff,
        CircuitBreaker,
    )

    @mcp.tool()
    @with_timeout(MEDIUM_TIMEOUT, "Database query timed out")
    async def query_database(query: str) -> dict:
        result = await db.execute(query)
        return asdict(success_response(data={"result": result}))
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import asyncio
import random
import time


# ---------------------------------------------------------------------------
# Timeout Budget Constants
# ---------------------------------------------------------------------------

#: Fast operations: cache lookups, simple queries (default 5s, max 10s)
FAST_TIMEOUT: float = 5.0
FAST_TIMEOUT_MAX: float = 10.0

#: Medium operations: database ops, API calls (default 30s, max 60s)
MEDIUM_TIMEOUT: float = 30.0
MEDIUM_TIMEOUT_MAX: float = 60.0

#: Slow operations: file processing, complex operations (default 120s, max 300s)
SLOW_TIMEOUT: float = 120.0
SLOW_TIMEOUT_MAX: float = 300.0

#: Background operations: batch jobs, large transfers (default 600s, max 3600s)
BACKGROUND_TIMEOUT: float = 600.0
BACKGROUND_TIMEOUT_MAX: float = 3600.0


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Timeout Error
# ---------------------------------------------------------------------------


class TimeoutException(Exception):
    """Operation timed out.

    Attributes:
        timeout_seconds: The timeout duration that was exceeded.
        operation: Name of the operation that timed out.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


# ---------------------------------------------------------------------------
# Timeout Decorator
# ---------------------------------------------------------------------------


def with_timeout(
    seconds: float,
    error_message: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add timeout to async functions.

    Uses asyncio.wait_for to enforce timeout on async operations.
    On timeout, raises TimeoutException with details.

    Args:
        seconds: Timeout duration in seconds.
        error_message: Custom error message (defaults to function name).

    Returns:
        Decorated async function with timeout enforcement.

    Example:
        >>> @with_timeout(30, "Database query timed out")
        ... async def query_database(query: str):
        ...     return await db.execute(query)

    Raises:
        TimeoutException: If the operation exceeds the timeout.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except asyncio.TimeoutError:
                msg = error_message or f"{func.__name__} timed out after {seconds}s"
                raise TimeoutException(
                    msg,
                    timeout_seconds=seconds,
                    operation=func.__name__,
                )

        return wrapper

    return decorator
