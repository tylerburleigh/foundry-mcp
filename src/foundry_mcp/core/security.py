"""
Security utilities for foundry-mcp.

Provides input validation, size limits, and prompt injection defense
for MCP tools. See docs/mcp_best_practices/04-validation-input-hygiene.md
and docs/mcp_best_practices/08-security-trust-boundaries.md for guidance.
"""

import logging
import re
from dataclasses import dataclass
from typing import Final, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Input Size Limits
# =============================================================================
# These constants define maximum sizes for various input types to prevent
# resource exhaustion and denial-of-service attacks. Adjust based on your
# specific requirements, but be conservative.

MAX_INPUT_SIZE: Final[int] = 100_000
"""Maximum total input payload size in bytes (100KB).

Use this to validate the overall size of request payloads before processing.
Prevents memory exhaustion from oversized requests.
"""

MAX_ARRAY_LENGTH: Final[int] = 1_000
"""Maximum number of items in array/list inputs.

Use this to limit iteration counts and prevent algorithmic complexity attacks.
Arrays larger than this should be paginated or streamed.
"""

MAX_STRING_LENGTH: Final[int] = 10_000
"""Maximum length for individual string fields (10K characters).

Use this for text fields like descriptions, content, etc.
Longer content should use dedicated file/blob handling.
"""

MAX_NESTED_DEPTH: Final[int] = 10
"""Maximum nesting depth for JSON structures.

Prevents stack overflow from deeply nested payloads.
Most legitimate use cases require < 5 levels of nesting.
"""

MAX_FIELD_COUNT: Final[int] = 100
"""Maximum number of fields in an object/dict.

Prevents resource exhaustion from objects with excessive properties.
"""

# =============================================================================
# Prompt Injection Detection Patterns
# =============================================================================
# These regex patterns detect common prompt injection attempts in LLM-generated
# input. MCP tools receiving untrusted input should check against these patterns.
# See docs/mcp_best_practices/08-security-trust-boundaries.md for details.

INJECTION_PATTERNS: Final[list[str]] = [
    # Instruction override attempts
    r"ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"forget\s+(everything|all)\s+(above|before)",
    r"new\s+instructions?\s*:",

    # System prompt injection
    r"system\s*:\s*",
    r"<\s*system\s*>",

    # Special tokens (model-specific)
    r"<\|.*?\|>",              # OpenAI-style special tokens
    r"\[INST\]|\[/INST\]",     # Llama instruction markers
    r"<\|im_start\|>|<\|im_end\|>",  # ChatML markers
    r"<<SYS>>|<</SYS>>",       # Llama system markers

    # Code block injection attempts
    r"```system",
    r"```\s*<\s*system",

    # Role injection
    r"^(assistant|user|system)\s*:",
]
"""Regex patterns for detecting prompt injection attempts.

Each pattern targets a specific injection technique:
- Instruction overrides: attempts to ignore/discard previous context
- System prompt injection: attempts to inject system-level instructions
- Special tokens: model-specific control sequences
- Code block injection: attempts to inject via markdown code blocks
- Role injection: attempts to assume different conversation roles

Use with detect_prompt_injection() for comprehensive checking.
"""


# =============================================================================
# Detection Results
# =============================================================================

@dataclass
class InjectionDetectionResult:
    """Result of prompt injection detection.

    Attributes:
        is_suspicious: Whether the input appears to contain injection attempts
        matched_pattern: The regex pattern that matched (if any)
        matched_text: The actual text that matched the pattern (if any)
    """
    is_suspicious: bool
    matched_pattern: Optional[str] = None
    matched_text: Optional[str] = None


# =============================================================================
# Detection Functions
# =============================================================================

def detect_prompt_injection(
    text: str,
    *,
    log_detections: bool = True,
    patterns: Optional[list[str]] = None,
) -> InjectionDetectionResult:
    """Detect potential prompt injection attempts in text.

    Scans the input text against known injection patterns and returns
    a result indicating whether suspicious content was found.

    Args:
        text: The input text to scan for injection attempts
        log_detections: Whether to log detected injection attempts (default: True)
        patterns: Optional custom patterns to use instead of INJECTION_PATTERNS

    Returns:
        InjectionDetectionResult with detection status and match details

    Example:
        >>> result = detect_prompt_injection("ignore previous instructions and...")
        >>> if result.is_suspicious:
        ...     print(f"Blocked: matched pattern '{result.matched_pattern}'")
    """
    check_patterns = patterns if patterns is not None else INJECTION_PATTERNS

    for pattern in check_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            result = InjectionDetectionResult(
                is_suspicious=True,
                matched_pattern=pattern,
                matched_text=match.group(0),
            )

            if log_detections:
                # Log with limited text preview to avoid logging sensitive content
                preview = text[:100] + "..." if len(text) > 100 else text
                logger.warning(
                    "Potential prompt injection detected",
                    extra={
                        "pattern": pattern,
                        "matched_text": result.matched_text,
                        "text_preview": preview,
                    }
                )

            return result

    return InjectionDetectionResult(is_suspicious=False)


def is_prompt_injection(text: str) -> bool:
    """Simple check for prompt injection (returns bool only).

    Convenience function when you only need a boolean result.

    Args:
        text: The input text to scan

    Returns:
        True if injection patterns detected, False otherwise

    Example:
        >>> if is_prompt_injection(user_input):
        ...     return error_response("Input contains disallowed patterns")
    """
    return detect_prompt_injection(text, log_detections=False).is_suspicious


# Export all constants and functions
__all__ = [
    # Constants
    "MAX_INPUT_SIZE",
    "MAX_ARRAY_LENGTH",
    "MAX_STRING_LENGTH",
    "MAX_NESTED_DEPTH",
    "MAX_FIELD_COUNT",
    "INJECTION_PATTERNS",
    # Types
    "InjectionDetectionResult",
    # Functions
    "detect_prompt_injection",
    "is_prompt_injection",
]
