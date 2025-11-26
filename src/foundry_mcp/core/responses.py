"""
Standard response contracts for MCP tool operations.
Provides consistent response structures across all foundry-mcp tools.

Response Schema Contract
========================

All MCP tool responses follow a standard structure:

    {
        "success": bool,       # Required: operation success/failure
        "data": {...},         # Required: primary payload (empty dict on error)
        "error": str | null,   # Required: error message or null on success
        "meta": {              # Required: response metadata
            "version": "response-v2"
        }
    }

Meta Payload Convention
-----------------------

For operations returning metadata alongside primary data, use reserved keys
within the data dict:

    data = {
        # Primary operation-specific fields
        "spec_id": "...",
        "tasks": [...],

        # Optional meta fields (reserved keys)
        "_meta": {
            "version": "1.0",           # API/schema version
            "pagination": {             # For paginated results
                "offset": 0,
                "limit": 50,
                "total": 150,
                "has_more": True
            },
            "timing": {                 # Performance info
                "duration_ms": 42
            }
        },
        "_warnings": [                  # Non-fatal issues
            "Spec has validation warnings",
            "Deprecated field 'foo' used"
        ]
    }

Multi-Payload Tools
-------------------

Tools returning multiple distinct payloads should nest them under named keys:

    data = {
        "spec": {...},          # First payload
        "tasks": [...],         # Second payload
        "_meta": {...}          # Metadata about the operation
    }

This ensures consumers can access each payload by key rather than relying
on position or implicit structure.

Edge Cases & Partial Payloads
-----------------------------

Empty Results (success=True):
    When a query succeeds but finds no results, return success=True with
    empty/partial data to distinguish from errors:

    # No tasks found (valid query, empty result)
    {"success": True, "data": {"tasks": [], "count": 0}, "error": None}

    # Spec complete (no more tasks to do)
    {"success": True, "data": {"found": False, "spec_complete": True}, "error": None}

Not Found (success=False):
    When the requested resource doesn't exist, return success=False:

    # Spec not found
    {"success": False, "data": {}, "error": "Spec not found: my-spec"}

    # Task not found
    {"success": False, "data": {}, "error": "Task not found: task-1-1"}

Blocked/Conditional States (success=True):
    Dependency checks and similar queries return success=True with state info:

    # Task is blocked but query succeeded
    {
        "success": True,
        "data": {
            "task_id": "task-1-2",
            "can_start": False,
            "blocked_by": [{"id": "task-1-1", "status": "pending"}]
        },
        "error": None
    }

Partial Success with Warnings:
    Operations that complete with caveats use _warnings:

    {
        "success": True,
        "data": {
            "validated": True,
            "spec_id": "my-spec",
            "_warnings": ["Field 'foo' is deprecated", "Missing optional metadata"]
        },
        "error": None
    }

Key Principle:
    - success=True means the operation executed correctly (even if result is empty)
    - success=False means the operation failed to execute
    - Use data fields (found, can_start, etc.) to convey semantic state
    - Use _warnings for non-fatal issues that don't prevent success
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResponse:
    """
    Standard response structure for MCP tool operations.

    All tool handlers should return data that can be serialized to this format,
    ensuring consistent API responses across the codebase.

    Attributes:
        success: Whether the operation completed successfully
        data: The primary payload (operation-specific structured data)
        error: Error message if success is False, None otherwise
        meta: Response metadata including version identifier
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=lambda: {"version": "response-v2"})


def success_response(**data: Any) -> ToolResponse:
    """
    Create a successful response with the given data.

    Args:
        **data: Keyword arguments to include in the response data

    Returns:
        ToolResponse with success=True and provided data

    Example:
        >>> success_response(spec_id="my-spec", count=5)
        ToolResponse(success=True, data={'spec_id': 'my-spec', 'count': 5}, error=None, meta={'version': 'response-v2'})
    """
    return ToolResponse(success=True, data=dict(data), error=None)


def error_response(message: str) -> ToolResponse:
    """
    Create an error response with the given message.

    Args:
        message: Error message describing what went wrong

    Returns:
        ToolResponse with success=False and error message

    Example:
        >>> error_response("Spec not found")
        ToolResponse(success=False, data={}, error='Spec not found', meta={'version': 'response-v2'})
    """
    return ToolResponse(success=False, data={}, error=message)
