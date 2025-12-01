"""
Base provider abstractions for foundry-mcp.

This module provides the core provider contracts adapted from sdd-toolkit,
enabling pluggable LLM backends for CLI operations. The abstractions support
capability negotiation, request/response normalization, and lifecycle hooks.

Design principles:
- Frozen dataclasses for immutability
- Enum-based capabilities for type-safe routing
- Status codes aligned with existing ProviderStatus patterns
- Error hierarchy for granular exception handling
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set

if TYPE_CHECKING:
    from foundry_mcp.core.responses import ErrorType


class ProviderCapability(Enum):
    """
    Feature flags a provider can expose to routing heuristics.

    These capabilities enable callers to select providers based on
    required features (vision, streaming, etc.) and allow registries
    to route requests to appropriate backends.

    Values:
        TEXT: Basic text generation capability
        VISION: Image/vision input processing
        FUNCTION_CALLING: Tool/function invocation support
        STREAMING: Incremental response streaming
        THINKING: Extended reasoning/chain-of-thought support
    """

    TEXT = "text_generation"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    THINKING = "thinking"


class ProviderStatus(Enum):
    """
    Normalized execution outcomes emitted by providers.

    These status codes provide a consistent interface for handling provider
    responses across different backends, enabling uniform error handling
    and retry logic.

    Values:
        SUCCESS: Operation completed successfully
        TIMEOUT: Operation exceeded time limit (retryable)
        NOT_FOUND: Provider/resource not available (not retryable)
        INVALID_OUTPUT: Provider returned malformed response (not retryable)
        ERROR: Generic error during execution (retryable)
        CANCELED: Operation was explicitly canceled (not retryable)
    """

    SUCCESS = "success"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    INVALID_OUTPUT = "invalid_output"
    ERROR = "error"
    CANCELED = "canceled"

    def is_retryable(self) -> bool:
        """
        Check if this status represents a transient failure that may succeed on retry.

        Retryable statuses:
            - TIMEOUT: May succeed with more time or if temporary resource contention resolves
            - ERROR: Generic errors may be transient (network issues, rate limits, etc.)

        Non-retryable statuses:
            - SUCCESS: Already succeeded
            - NOT_FOUND: Provider/tool not available (won't change on retry)
            - INVALID_OUTPUT: Provider responded but output was malformed
            - CANCELED: Explicitly canceled (shouldn't retry)

        Returns:
            True if the status is retryable, False otherwise
        """
        return self in (ProviderStatus.TIMEOUT, ProviderStatus.ERROR)

    def to_error_type(self) -> "ErrorType":
        """
        Map provider status to foundry-mcp ErrorType for response envelopes.

        This enables consistent error categorization across MCP and CLI surfaces,
        allowing callers to handle provider errors using the standard error taxonomy.

        Mapping:
            - SUCCESS: raises ValueError (not an error state)
            - TIMEOUT: UNAVAILABLE (503 analog, retryable)
            - NOT_FOUND: NOT_FOUND (404 analog, not retryable)
            - INVALID_OUTPUT: VALIDATION (400 analog, not retryable)
            - ERROR: INTERNAL (500 analog, retryable)
            - CANCELED: INTERNAL (operation aborted)

        Returns:
            ErrorType enum value corresponding to this status

        Raises:
            ValueError: If called on SUCCESS status (not an error)
        """
        from foundry_mcp.core.responses import ErrorType

        if self == ProviderStatus.SUCCESS:
            raise ValueError("SUCCESS status cannot be mapped to an error type")

        mapping = {
            ProviderStatus.TIMEOUT: ErrorType.UNAVAILABLE,
            ProviderStatus.NOT_FOUND: ErrorType.NOT_FOUND,
            ProviderStatus.INVALID_OUTPUT: ErrorType.VALIDATION,
            ProviderStatus.ERROR: ErrorType.INTERNAL,
            ProviderStatus.CANCELED: ErrorType.INTERNAL,
        }
        return mapping[self]


@dataclass(frozen=True)
class ProviderRequest:
    """
    Normalized request payload for provider execution.

    This dataclass encapsulates all parameters needed to make a generation
    request to any provider backend. Fields follow common LLM API conventions
    to ensure portability across different providers.

    Attributes:
        prompt: The user's input prompt/message
        system_prompt: Optional system/instruction prompt
        model: Model identifier (provider-specific, e.g., "pro", "flash")
        timeout: Request timeout in seconds (None = provider default)
        temperature: Sampling temperature (0.0-2.0, None = provider default)
        max_tokens: Maximum output tokens (None = provider default)
        metadata: Arbitrary request metadata (tracing IDs, feature flags, etc.)
        stream: Whether to request streaming response
        attachments: File paths or URIs for multimodal inputs
    """

    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    timeout: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    attachments: Sequence[str] = field(default_factory=list)
