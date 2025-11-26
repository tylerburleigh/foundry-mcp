"""
Test harness infrastructure for parity testing.

Provides adapters, normalizers, and comparators for testing
foundry-mcp and sdd-toolkit equivalence.
"""

from .base import SpecToolAdapter
from .foundry_adapter import FoundryMcpAdapter
from .sdd_adapter import SddToolkitAdapter
from .normalizers import OutputNormalizer, FieldMapper
from .comparators import ResultComparator

__all__ = [
    "SpecToolAdapter",
    "FoundryMcpAdapter",
    "SddToolkitAdapter",
    "OutputNormalizer",
    "FieldMapper",
    "ResultComparator",
]
