"""
Result comparison utilities for parity testing.

Provides deep comparison of normalized outputs with detailed diff output.
"""

import json
from difflib import unified_diff
from typing import Any, Dict, Optional, Set, Tuple


class ResultComparator:
    """Compare results from both systems with detailed diff output."""

    @classmethod
    def compare(
        cls,
        foundry_result: Dict[str, Any],
        sdd_result: Dict[str, Any],
        ignore_fields: Optional[Set[str]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Compare normalized results from both systems.

        Args:
            foundry_result: Normalized result from foundry-mcp
            sdd_result: Normalized result from sdd-toolkit
            ignore_fields: Fields to ignore in comparison

        Returns:
            Tuple of (is_equal, diff_string or None)
        """
        ignore = ignore_fields or set()

        # Filter out ignored fields
        foundry_filtered = cls._filter_fields(foundry_result, ignore)
        sdd_filtered = cls._filter_fields(sdd_result, ignore)

        # Compare
        if foundry_filtered == sdd_filtered:
            return True, None

        # Generate diff
        foundry_json = json.dumps(foundry_filtered, indent=2, sort_keys=True)
        sdd_json = json.dumps(sdd_filtered, indent=2, sort_keys=True)

        diff = "\n".join(
            unified_diff(
                foundry_json.splitlines(),
                sdd_json.splitlines(),
                fromfile="foundry-mcp",
                tofile="sdd-toolkit",
                lineterm="",
            )
        )

        return False, diff

    @classmethod
    def _filter_fields(
        cls, data: Any, ignore: Set[str]
    ) -> Any:
        """Recursively filter out ignored fields."""
        if isinstance(data, dict):
            return {
                k: cls._filter_fields(v, ignore)
                for k, v in data.items()
                if k not in ignore
            }
        elif isinstance(data, list):
            return [cls._filter_fields(item, ignore) for item in data]
        return data

    @classmethod
    def assert_parity(
        cls,
        foundry_result: Dict[str, Any],
        sdd_result: Dict[str, Any],
        operation: str,
        ignore_fields: Optional[Set[str]] = None,
    ) -> None:
        """
        Assert that results match, with detailed failure message.

        Args:
            foundry_result: Normalized result from foundry-mcp
            sdd_result: Normalized result from sdd-toolkit
            operation: Name of operation being tested (for error message)
            ignore_fields: Fields to ignore in comparison

        Raises:
            AssertionError: If results don't match
        """
        is_equal, diff = cls.compare(foundry_result, sdd_result, ignore_fields)

        if not is_equal:
            message = (
                f"\n{'=' * 60}\n"
                f"Parity failure for operation: {operation}\n"
                f"{'=' * 60}\n"
                f"{diff}"
            )
            raise AssertionError(message)

    @classmethod
    def assert_key_match(
        cls,
        foundry_result: Dict[str, Any],
        sdd_result: Dict[str, Any],
        key: str,
        operation: str,
    ) -> None:
        """
        Assert that a specific key matches between results.

        Args:
            foundry_result: Result from foundry-mcp
            sdd_result: Result from sdd-toolkit
            key: Key to compare
            operation: Name of operation being tested

        Raises:
            AssertionError: If key values don't match
        """
        foundry_val = foundry_result.get(key)
        sdd_val = sdd_result.get(key)

        if foundry_val != sdd_val:
            raise AssertionError(
                f"Key mismatch for '{key}' in {operation}:\n"
                f"  foundry-mcp: {foundry_val}\n"
                f"  sdd-toolkit: {sdd_val}"
            )

    @classmethod
    def assert_success(
        cls,
        foundry_result: Dict[str, Any],
        sdd_result: Dict[str, Any],
        operation: str,
    ) -> None:
        """
        Assert that both operations succeeded.

        Args:
            foundry_result: Result from foundry-mcp
            sdd_result: Result from sdd-toolkit
            operation: Name of operation being tested

        Raises:
            AssertionError: If either operation failed
        """
        foundry_success = foundry_result.get("success", True)
        sdd_success = sdd_result.get("success", True)

        # Also check for error keys
        foundry_has_error = "error" in foundry_result
        sdd_has_error = "error" in sdd_result

        if not foundry_success or foundry_has_error:
            raise AssertionError(
                f"foundry-mcp failed for {operation}: "
                f"{foundry_result.get('error', 'Unknown error')}"
            )

        if not sdd_success or sdd_has_error:
            raise AssertionError(
                f"sdd-toolkit failed for {operation}: "
                f"{sdd_result.get('error', 'Unknown error')}"
            )

    @classmethod
    def assert_both_error(
        cls,
        foundry_result: Dict[str, Any],
        sdd_result: Dict[str, Any],
        operation: str,
    ) -> None:
        """
        Assert that both operations returned an error.

        Args:
            foundry_result: Result from foundry-mcp
            sdd_result: Result from sdd-toolkit
            operation: Name of operation being tested

        Raises:
            AssertionError: If one succeeded and other failed
        """
        foundry_error = (
            foundry_result.get("success") is False
            or "error" in foundry_result
        )
        sdd_error = (
            sdd_result.get("success") is False
            or "error" in sdd_result
        )

        if foundry_error != sdd_error:
            raise AssertionError(
                f"Error status mismatch for {operation}:\n"
                f"  foundry-mcp error: {foundry_error}\n"
                f"  sdd-toolkit error: {sdd_error}"
            )
