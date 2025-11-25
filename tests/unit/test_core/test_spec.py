"""Tests for core spec operations."""

import json
import tempfile
from pathlib import Path

import pytest

from foundry_mcp.core.spec import (
    find_specs_directory,
    find_spec_file,
    load_spec,
    save_spec,
    list_specs,
    get_node,
    update_node,
)


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Resolve to handle macOS /var -> /private/var symlink
        specs_dir = (Path(tmpdir) / "specs").resolve()

        # Create status directories
        (specs_dir / "pending").mkdir(parents=True)
        (specs_dir / "active").mkdir(parents=True)
        (specs_dir / "completed").mkdir(parents=True)
        (specs_dir / "archived").mkdir(parents=True)

        yield specs_dir


@pytest.fixture
def sample_spec():
    """Create a sample spec data structure."""
    return {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "metadata": {
            "title": "Test Specification",
            "version": "1.0.0",
        },
        "hierarchy": {
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "pending",
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "pending",
                "parent": "phase-1",
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 2",
                "status": "completed",
                "parent": "phase-1",
            },
        }
    }


class TestFindSpecsDirectory:
    """Tests for find_specs_directory function."""

    def test_find_specs_directory_with_explicit_path(self, temp_specs_dir):
        """Should find specs directory when given explicit path."""
        result = find_specs_directory(str(temp_specs_dir))
        assert result == temp_specs_dir

    def test_find_specs_directory_from_parent(self, temp_specs_dir):
        """Should find specs directory from parent path."""
        parent = temp_specs_dir.parent
        result = find_specs_directory(str(parent))
        assert result == temp_specs_dir


class TestFindSpecFile:
    """Tests for find_spec_file function."""

    def test_find_spec_in_active(self, temp_specs_dir, sample_spec):
        """Should find spec in active folder."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result = find_spec_file("test-spec-001", temp_specs_dir)
        assert result == spec_file

    def test_find_spec_in_pending(self, temp_specs_dir, sample_spec):
        """Should find spec in pending folder."""
        spec_file = temp_specs_dir / "pending" / "test-spec-002.json"
        sample_spec["spec_id"] = "test-spec-002"
        spec_file.write_text(json.dumps(sample_spec))

        result = find_spec_file("test-spec-002", temp_specs_dir)
        assert result == spec_file

    def test_spec_not_found(self, temp_specs_dir):
        """Should return None when spec not found."""
        result = find_spec_file("nonexistent-spec", temp_specs_dir)
        assert result is None


class TestLoadSpec:
    """Tests for load_spec function."""

    def test_load_spec_success(self, temp_specs_dir, sample_spec):
        """Should load spec successfully."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result = load_spec("test-spec-001", temp_specs_dir)
        assert result is not None
        assert result["spec_id"] == "test-spec-001"
        assert result["title"] == "Test Specification"

    def test_load_spec_not_found(self, temp_specs_dir):
        """Should return None for nonexistent spec."""
        result = load_spec("nonexistent-spec", temp_specs_dir)
        assert result is None


class TestListSpecs:
    """Tests for list_specs function."""

    def test_list_all_specs(self, temp_specs_dir, sample_spec):
        """Should list all specs across folders."""
        # Create specs in different folders
        (temp_specs_dir / "active" / "spec-1.json").write_text(
            json.dumps({**sample_spec, "spec_id": "spec-1"})
        )
        (temp_specs_dir / "pending" / "spec-2.json").write_text(
            json.dumps({**sample_spec, "spec_id": "spec-2"})
        )

        result = list_specs(specs_dir=temp_specs_dir)
        assert len(result) == 2
        spec_ids = [s["spec_id"] for s in result]
        assert "spec-1" in spec_ids
        assert "spec-2" in spec_ids

    def test_list_specs_by_status(self, temp_specs_dir, sample_spec):
        """Should filter specs by status."""
        (temp_specs_dir / "active" / "spec-1.json").write_text(
            json.dumps({**sample_spec, "spec_id": "spec-1"})
        )
        (temp_specs_dir / "pending" / "spec-2.json").write_text(
            json.dumps({**sample_spec, "spec_id": "spec-2"})
        )

        result = list_specs(specs_dir=temp_specs_dir, status="active")
        assert len(result) == 1
        assert result[0]["spec_id"] == "spec-1"


class TestGetNode:
    """Tests for get_node function."""

    def test_get_existing_node(self, sample_spec):
        """Should return node data for existing node."""
        result = get_node(sample_spec, "task-1-1")
        assert result is not None
        assert result["title"] == "Task 1"
        assert result["status"] == "pending"

    def test_get_nonexistent_node(self, sample_spec):
        """Should return None for nonexistent node."""
        result = get_node(sample_spec, "nonexistent")
        assert result is None


class TestUpdateNode:
    """Tests for update_node function."""

    def test_update_existing_node(self, sample_spec):
        """Should update node and return True."""
        result = update_node(sample_spec, "task-1-1", {"status": "in_progress"})
        assert result is True
        assert sample_spec["hierarchy"]["task-1-1"]["status"] == "in_progress"

    def test_update_nonexistent_node(self, sample_spec):
        """Should return False for nonexistent node."""
        result = update_node(sample_spec, "nonexistent", {"status": "completed"})
        assert result is False

    def test_update_preserves_existing_fields(self, sample_spec):
        """Should preserve fields not being updated."""
        result = update_node(sample_spec, "task-1-1", {"status": "completed"})
        assert result is True
        assert sample_spec["hierarchy"]["task-1-1"]["title"] == "Task 1"
        assert sample_spec["hierarchy"]["task-1-1"]["parent"] == "phase-1"
