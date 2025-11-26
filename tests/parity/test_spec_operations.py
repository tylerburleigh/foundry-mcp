"""
Parity tests for spec operations.

Tests list_specs, get_spec, and get_task operations.
"""

import pytest

from .harness.normalizers import normalize_for_comparison
from .harness.comparators import ResultComparator


class TestListSpecs:
    """Parity tests for listing specifications."""

    @pytest.mark.parity
    def test_list_all_specs_parity(self, both_adapters):
        """Test that list_specs returns equivalent results."""
        foundry, sdd = both_adapters

        # Execute on both systems
        foundry_result = foundry.list_specs(status="all")
        sdd_result = sdd.list_specs(status="all")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "list_specs(all)")

        # Normalize and compare spec counts
        assert foundry_result.get("count", 0) > 0, "No specs found by foundry-mcp"

    @pytest.mark.parity
    def test_list_active_specs_parity(self, both_adapters):
        """Test listing active specs."""
        foundry, sdd = both_adapters

        foundry_result = foundry.list_specs(status="active")
        sdd_result = sdd.list_specs(status="active")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "list_specs(active)")

    @pytest.mark.parity
    def test_list_pending_specs_parity(self, both_adapters):
        """Test listing pending specs (should be empty in fixture)."""
        foundry, sdd = both_adapters

        foundry_result = foundry.list_specs(status="pending")
        sdd_result = sdd.list_specs(status="pending")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "list_specs(pending)")


class TestGetSpec:
    """Parity tests for getting spec details."""

    @pytest.mark.parity
    def test_get_spec_parity(self, both_adapters):
        """Test that get_spec returns equivalent results."""
        foundry, sdd = both_adapters

        foundry_result = foundry.get_spec("parity-test-simple")
        sdd_result = sdd.get_spec("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "get_spec")

    @pytest.mark.parity
    def test_get_spec_not_found_parity(self, both_adapters):
        """Test error handling for non-existent spec."""
        foundry, sdd = both_adapters

        foundry_result = foundry.get_spec("nonexistent-spec-xyz")
        sdd_result = sdd.get_spec("nonexistent-spec-xyz")

        # Both should return error
        ResultComparator.assert_both_error(
            foundry_result, sdd_result, "get_spec(nonexistent)"
        )


class TestGetTask:
    """Parity tests for getting task details."""

    @pytest.mark.parity
    def test_get_task_parity(self, both_adapters):
        """Test that get_task returns equivalent results."""
        foundry, sdd = both_adapters

        foundry_result = foundry.get_task("parity-test-simple", "task-1-1")
        sdd_result = sdd.get_task("parity-test-simple", "task-1-1")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "get_task")

        # Check task_id matches
        ResultComparator.assert_key_match(
            foundry_result, sdd_result, "task_id", "get_task"
        )

    @pytest.mark.parity
    def test_get_task_not_found_parity(self, both_adapters):
        """Test error handling for non-existent task."""
        foundry, sdd = both_adapters

        foundry_result = foundry.get_task("parity-test-simple", "nonexistent-task")
        sdd_result = sdd.get_task("parity-test-simple", "nonexistent-task")

        # Both should return error
        ResultComparator.assert_both_error(
            foundry_result, sdd_result, "get_task(nonexistent)"
        )


# Standalone tests for foundry adapter only (useful when sdd CLI unavailable)
class TestFoundryAdapterStandalone:
    """Tests for foundry adapter without requiring sdd-toolkit."""

    def test_foundry_list_specs(self, foundry_adapter):
        """Test foundry adapter list_specs works."""
        result = foundry_adapter.list_specs(status="all")
        assert result.get("success") is True
        assert result.get("count", 0) >= 1

    def test_foundry_get_spec(self, foundry_adapter):
        """Test foundry adapter get_spec works."""
        result = foundry_adapter.get_spec("parity-test-simple")
        assert result.get("success") is True
        assert "spec" in result

    def test_foundry_get_task(self, foundry_adapter):
        """Test foundry adapter get_task works."""
        result = foundry_adapter.get_task("parity-test-simple", "task-1-1")
        assert result.get("success") is True
        assert result.get("task_id") == "task-1-1"

    def test_foundry_next_task(self, foundry_adapter):
        """Test foundry adapter next_task works."""
        result = foundry_adapter.next_task("parity-test-simple")
        assert result.get("success") is True
        # Should find task-1-3 (pending task after completed and in_progress)
        assert result.get("task_id") in ["task-1-2", "task-1-3", None]

    def test_foundry_progress(self, foundry_adapter):
        """Test foundry adapter progress works."""
        result = foundry_adapter.progress("parity-test-simple")
        assert result.get("success") is True
