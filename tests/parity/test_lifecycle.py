"""
Parity tests for spec lifecycle operations.

Tests activate_spec, complete_spec, archive_spec, and move_spec operations.
"""

import pytest

from .harness.comparators import ResultComparator
from .harness.fixture_manager import FixtureManager
from .harness.foundry_adapter import FoundryMcpAdapter
from .harness.sdd_adapter import SddToolkitAdapter


class TestActivateSpec:
    """Parity tests for activating specs."""

    @pytest.mark.parity
    def test_activate_spec_parity(self, test_dir):
        """Test that activate_spec produces equivalent results."""
        # Setup foundry copy with spec in pending
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("simple_spec", status="pending")
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        # Setup sdd copy with spec in pending
        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("simple_spec", status="pending")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Activate on both
        foundry_result = foundry.activate_spec("parity-test-simple")
        sdd_result = sdd.activate_spec("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "activate_spec")

        # Verify spec moved to active folder
        assert (foundry_root / "specs" / "active" / "parity-test-simple.json").exists()
        assert not (foundry_root / "specs" / "pending" / "parity-test-simple.json").exists()

    @pytest.mark.parity
    def test_activate_nonexistent_spec_parity(self, test_dir):
        """Test error handling when activating non-existent spec."""
        # Setup empty fixtures
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("simple_spec", status="active")  # Different spec
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("simple_spec", status="active")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Try to activate non-existent spec
        foundry_result = foundry.activate_spec("nonexistent-spec-xyz")
        sdd_result = sdd.activate_spec("nonexistent-spec-xyz")

        # Both should return error
        ResultComparator.assert_both_error(
            foundry_result, sdd_result, "activate_spec(nonexistent)"
        )


class TestCompleteSpec:
    """Parity tests for completing specs."""

    @pytest.mark.parity
    def test_complete_spec_parity(self, test_dir):
        """Test that complete_spec produces equivalent results."""
        # Use completed_spec fixture where all tasks are done
        # (sdd-toolkit requires all tasks complete before moving to completed)
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("completed_spec", status="active")
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("completed_spec", status="active")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Complete on both
        foundry_result = foundry.complete_spec("parity-test-completed")
        sdd_result = sdd.complete_spec("parity-test-completed")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "complete_spec")

        # Verify spec moved to completed folder
        assert (foundry_root / "specs" / "completed" / "parity-test-completed.json").exists()
        assert not (foundry_root / "specs" / "active" / "parity-test-completed.json").exists()

    @pytest.mark.parity
    def test_complete_nonexistent_spec_parity(self, test_dir):
        """Test error handling when completing non-existent spec."""
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("simple_spec", status="active")
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("simple_spec", status="active")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Try to complete non-existent spec
        foundry_result = foundry.complete_spec("nonexistent-spec-xyz")
        sdd_result = sdd.complete_spec("nonexistent-spec-xyz")

        # Both should return error
        ResultComparator.assert_both_error(
            foundry_result, sdd_result, "complete_spec(nonexistent)"
        )


class TestArchiveSpec:
    """Parity tests for archiving specs."""

    @pytest.mark.parity
    def test_archive_spec_parity(self, test_dir):
        """Test that archive_spec produces equivalent results."""
        # Setup foundry copy with spec in active
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("simple_spec", status="active")
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        # Setup sdd copy with spec in active
        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("simple_spec", status="active")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Archive on both
        foundry_result = foundry.archive_spec("parity-test-simple")
        sdd_result = sdd.archive_spec("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "archive_spec")

        # Verify spec moved to archived folder
        assert (foundry_root / "specs" / "archived" / "parity-test-simple.json").exists()
        assert not (foundry_root / "specs" / "active" / "parity-test-simple.json").exists()

    @pytest.mark.parity
    def test_archive_nonexistent_spec_parity(self, test_dir):
        """Test error handling when archiving non-existent spec."""
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("simple_spec", status="active")
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("simple_spec", status="active")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Try to archive non-existent spec
        foundry_result = foundry.archive_spec("nonexistent-spec-xyz")
        sdd_result = sdd.archive_spec("nonexistent-spec-xyz")

        # Both should return error
        ResultComparator.assert_both_error(
            foundry_result, sdd_result, "archive_spec(nonexistent)"
        )


class TestMoveSpec:
    """Parity tests for moving specs between status folders."""

    @pytest.mark.parity
    def test_move_spec_parity(self, test_dir):
        """Test that move_spec produces equivalent results."""
        # Setup foundry copy with spec in active
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("simple_spec", status="active")
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        # Setup sdd copy with spec in active
        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("simple_spec", status="active")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Move to completed on both (sdd-toolkit doesn't support moving to pending)
        foundry_result = foundry.move_spec("parity-test-simple", "completed")
        sdd_result = sdd.move_spec("parity-test-simple", "completed")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "move_spec")

        # Verify spec moved to completed folder
        assert (foundry_root / "specs" / "completed" / "parity-test-simple.json").exists()
        assert not (foundry_root / "specs" / "active" / "parity-test-simple.json").exists()

    @pytest.mark.parity
    def test_move_spec_to_archived_parity(self, test_dir):
        """Test moving spec directly to archived."""
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("simple_spec", status="pending")
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("simple_spec", status="pending")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Move directly to archived
        foundry_result = foundry.move_spec("parity-test-simple", "archived")
        sdd_result = sdd.move_spec("parity-test-simple", "archived")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "move_spec(to archived)")

    @pytest.mark.parity
    def test_move_nonexistent_spec_parity(self, test_dir):
        """Test error handling when moving non-existent spec."""
        foundry_root = test_dir / "foundry"
        foundry_fixture = FixtureManager(foundry_root)
        foundry_fixture.setup("simple_spec", status="active")
        foundry = FoundryMcpAdapter(foundry_root / "specs")

        sdd_root = test_dir / "sdd"
        sdd_fixture = FixtureManager(sdd_root)
        sdd_fixture.setup("simple_spec", status="active")
        sdd = SddToolkitAdapter(sdd_root / "specs")

        # Try to move non-existent spec
        foundry_result = foundry.move_spec("nonexistent-spec-xyz", "completed")
        sdd_result = sdd.move_spec("nonexistent-spec-xyz", "completed")

        # Both should return error
        ResultComparator.assert_both_error(
            foundry_result, sdd_result, "move_spec(nonexistent)"
        )


# Standalone tests for foundry adapter
class TestFoundryLifecycleOperations:
    """Tests for foundry adapter lifecycle operations."""

    def test_foundry_activate_spec(self, test_dir):
        """Test foundry adapter activate_spec works."""
        fixture = FixtureManager(test_dir)
        fixture.setup("simple_spec", status="pending")
        adapter = FoundryMcpAdapter(test_dir / "specs")

        result = adapter.activate_spec("parity-test-simple")
        assert result.get("success") is True
        assert result.get("new_status") == "active"

    def test_foundry_complete_spec(self, test_dir):
        """Test foundry adapter complete_spec works."""
        fixture = FixtureManager(test_dir)
        fixture.setup("simple_spec", status="active")
        adapter = FoundryMcpAdapter(test_dir / "specs")

        result = adapter.complete_spec("parity-test-simple")
        assert result.get("success") is True
        assert result.get("new_status") == "completed"

    def test_foundry_archive_spec(self, test_dir):
        """Test foundry adapter archive_spec works."""
        fixture = FixtureManager(test_dir)
        fixture.setup("simple_spec", status="active")
        adapter = FoundryMcpAdapter(test_dir / "specs")

        result = adapter.archive_spec("parity-test-simple")
        assert result.get("success") is True
        assert result.get("new_status") == "archived"

    def test_foundry_move_spec(self, test_dir):
        """Test foundry adapter move_spec works."""
        fixture = FixtureManager(test_dir)
        fixture.setup("simple_spec", status="active")
        adapter = FoundryMcpAdapter(test_dir / "specs")

        result = adapter.move_spec("parity-test-simple", "pending")
        assert result.get("success") is True
        assert result.get("new_status") == "pending"
