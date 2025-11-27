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


# ---------------------------------------------------------------------------
# Retry with Backoff
# ---------------------------------------------------------------------------


def retry_with_backoff(
    func: Callable[..., T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
) -> T:
    """Retry a function with exponential backoff.

    Retries the function on failure with increasing delays between attempts.
    Supports jitter to prevent thundering herd problems.

    Args:
        func: Function to retry (should take no arguments; use lambda for args).
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay in seconds (default 1.0).
        max_delay: Maximum delay cap in seconds (default 60.0).
        exponential_base: Multiplier for each retry (default 2.0).
        jitter: Add randomness to delay (default True).
        retryable_exceptions: List of exceptions to retry on (default: all).

    Returns:
        Result from the function on success.

    Raises:
        Exception: The last exception if all retries exhausted.

    Example:
        >>> result = retry_with_backoff(
        ...     lambda: http_client.get(url),
        ...     max_retries=3,
        ...     retryable_exceptions=[ConnectionError, TimeoutException],
        ... )
    """
    retryable = tuple(retryable_exceptions or [Exception])
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable as e:
            last_exception = e

            if attempt == max_retries:
                break

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base**attempt), max_delay)

            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random())

            time.sleep(delay)

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise RuntimeError("retry_with_backoff: unexpected state")


def retryable(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for automatic retries with exponential backoff.

    Args:
        max_retries: Maximum retry attempts (default 3).
        delay: Base delay in seconds (default 1.0).
        exceptions: Tuple of exceptions to retry on.

    Returns:
        Decorated function with retry logic.

    Example:
        >>> @retryable(max_retries=3, exceptions=(ConnectionError,))
        ... def call_api(endpoint: str):
        ...     return http_client.get(endpoint)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=delay,
                retryable_exceptions=list(exceptions),
            )

        return wrapper

    return decorator
