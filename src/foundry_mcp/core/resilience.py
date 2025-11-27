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
