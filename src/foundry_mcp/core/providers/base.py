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
from typing import Any, Callable, Dict, List, Optional, Sequence, Set


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
