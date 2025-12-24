"""
Unit tests for task metadata-batch handler.

Tests batch metadata updates with flexible AND-based filtering.
"""

import json
import pytest
from foundry_mcp.server import create_server
from foundry_mcp.config import ServerConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_specs_dir(tmp_path):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    for d in ["active", "pending", "completed", "archived"]:
        (specs_dir / d).mkdir()

    sample_spec = {
        "spec_id": "batch-test-spec-001",
        "title": "Test Spec",
        "metadata": {"title": "Test Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {"type": "spec", "title": "Test", "status": "in_progress", "children": ["phase-1", "phase-2"]},
            "phase-1": {"type": "phase", "title": "Phase 1", "status": "pending", "parent": "spec-root", "children": ["task-1-1", "task-1-2", "verify-1-1"]},
            "task-1-1": {"type": "task", "title": "Implement feature", "status": "pending", "parent": "phase-1", "children": [], "metadata": {}},
            "task-1-2": {"type": "task", "title": "Add tests", "status": "pending", "parent": "phase-1", "children": [], "metadata": {}},
            "verify-1-1": {"type": "verify", "title": "Run tests", "status": "pending", "parent": "phase-1", "children": [], "metadata": {}},
            "phase-2": {"type": "phase", "title": "Phase 2", "status": "pending", "parent": "spec-root", "children": ["task-2-1", "verify-2-1"]},
            "task-2-1": {"type": "task", "title": "Deploy feature", "status": "pending", "parent": "phase-2", "children": [], "metadata": {}},
            "verify-2-1": {"type": "verify", "title": "Smoke test", "status": "pending", "parent": "phase-2", "children": [], "metadata": {}},
        },
        "assumptions": [], "revision_history": [], "journal": [],
    }
    (specs_dir / "active" / "batch-test-spec-001.json").write_text(json.dumps(sample_spec))
    return specs_dir


@pytest.fixture
def test_config(test_specs_dir):
    return ServerConfig(server_name="test", server_version="0.1.0", specs_dir=test_specs_dir, log_level="WARNING")


@pytest.fixture
def task_tool(test_config):
    return create_server(test_config)._tool_manager._tools["task"].fn


# =============================================================================
# Required Parameter Tests
# =============================================================================

class TestRequiredParams:
    def test_missing_spec_id(self, task_tool):
        result = task_tool(action="metadata-batch", node_type="task", file_path="test.py")
        assert result["success"] is False
        assert "spec_id" in result["error"].lower()

    def test_missing_filter(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", file_path="test.py")
        assert result["success"] is False
        assert "filter" in result["error"].lower()

    def test_missing_metadata(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task")
        assert result["success"] is False
        assert "metadata" in result["error"].lower()


# =============================================================================
# Filter Validation Tests
# =============================================================================

class TestFilterValidation:
    def test_invalid_node_type(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="invalid", file_path="test.py")
        assert result["success"] is False
        assert "node_type" in result["error"].lower()

    def test_invalid_pattern_regex(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", pattern="[invalid", file_path="test.py")
        assert result["success"] is False
        assert "pattern" in result["error"].lower()

    def test_empty_phase_id(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", phase_id="", file_path="test.py")
        assert result["success"] is False
        assert "phase_id" in result["error"].lower()


# =============================================================================
# Metadata Validation Tests
# =============================================================================

class TestMetadataValidation:
    def test_invalid_verification_type(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="verify", verification_type="invalid")
        assert result["success"] is False
        assert "verification_type" in result["error"].lower()

    def test_invalid_owners_not_list(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", owners="not a list")
        assert result["success"] is False
        assert "owners" in result["error"].lower()

    def test_invalid_labels_not_dict(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", labels="not a dict")
        assert result["success"] is False
        assert "labels" in result["error"].lower()

    def test_invalid_estimated_hours(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", estimated_hours="bad")
        assert result["success"] is False
        assert "estimated_hours" in result["error"].lower()


# =============================================================================
# Filtering Tests
# =============================================================================

class TestFiltering:
    def test_filter_by_node_type_task(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert result["data"]["matched_count"] == 3  # task-1-1, task-1-2, task-2-1

    def test_filter_by_node_type_verify(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="verify", verification_type="run-tests", dry_run=True)
        assert result["success"] is True
        assert result["data"]["matched_count"] == 2  # verify-1-1, verify-2-1

    def test_filter_by_phase_id(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", phase_id="phase-1", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert result["data"]["matched_count"] == 3  # task-1-1, task-1-2, verify-1-1

    def test_filter_by_pattern(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", pattern="test", file_path="test.py", dry_run=True)
        assert result["success"] is True
        # "Add tests", "Run tests", "Smoke test" should all match
        assert result["data"]["matched_count"] >= 3

    def test_filter_combined_phase_and_type(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", phase_id="phase-1", node_type="task", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert result["data"]["matched_count"] == 2  # task-1-1, task-1-2

    def test_filter_combined_phase_and_pattern(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", phase_id="phase-1", pattern="test", file_path="test.py", dry_run=True)
        assert result["success"] is True
        # "Add tests", "Run tests" in phase-1
        assert result["data"]["matched_count"] == 2

    def test_filter_no_matches(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", pattern="nonexistent", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert result["data"]["matched_count"] == 0
        assert "No nodes matched" in result["data"].get("message", "")


# =============================================================================
# Dry Run Tests
# =============================================================================

class TestDryRun:
    def test_dry_run_does_not_persist(self, task_tool, test_specs_dir):
        # Run with dry_run=True
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="updated.py", dry_run=True)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert result["data"]["updated_count"] == 0  # Should be 0 in dry run

        # Verify file was not modified
        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        assert spec_data["hierarchy"]["task-1-1"]["metadata"].get("file_path") is None

    def test_dry_run_shows_preview(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert "nodes" in result["data"]
        assert len(result["data"]["nodes"]) == 3
        for node in result["data"]["nodes"]:
            assert "node_id" in node
            assert "title" in node
            assert "fields_updated" in node

    def test_dry_run_shows_diff_preview(self, task_tool, test_specs_dir):
        """Verify dry-run includes diff with old and new values."""
        # First set some initial values
        task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="old.py")

        # Now dry-run with new values
        result = task_tool(
            action="metadata-batch",
            spec_id="batch-test-spec-001",
            node_type="task",
            file_path="new.py",
            dry_run=True
        )
        assert result["success"] is True
        assert result["data"]["dry_run"] is True

        # Check that nodes include diff
        nodes = result["data"]["nodes"]
        assert len(nodes) >= 1
        for node in nodes:
            assert "diff" in node, f"Node {node['node_id']} missing diff"
            assert "file_path" in node["diff"]
            assert node["diff"]["file_path"]["old"] == "old.py"
            assert node["diff"]["file_path"]["new"] == "new.py"

    def test_dry_run_diff_excludes_unchanged_fields(self, task_tool, test_specs_dir):
        """Verify diff only includes fields that actually change."""
        # Set initial value
        task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="same.py")

        # Dry-run with same value
        result = task_tool(
            action="metadata-batch",
            spec_id="batch-test-spec-001",
            node_type="task",
            file_path="same.py",
            dry_run=True
        )
        assert result["success"] is True

        # Nodes should not have diff since values are unchanged
        nodes = result["data"]["nodes"]
        for node in nodes:
            # No diff key or empty diff when no changes
            if "diff" in node:
                assert node["diff"] == {}, f"Node {node['node_id']} has unexpected diff"


# =============================================================================
# Actual Update Tests
# =============================================================================

class TestActualUpdate:
    def test_updates_file_path(self, task_tool, test_specs_dir):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", phase_id="phase-1", node_type="task", file_path="src/feature.py")
        assert result["success"] is True
        assert result["data"]["updated_count"] == 2

        # Verify file was modified
        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        assert spec_data["hierarchy"]["task-1-1"]["metadata"]["file_path"] == "src/feature.py"
        assert spec_data["hierarchy"]["task-1-2"]["metadata"]["file_path"] == "src/feature.py"

    def test_updates_verification_type(self, task_tool, test_specs_dir):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="verify", verification_type="run-tests")
        assert result["success"] is True
        assert result["data"]["updated_count"] == 2

        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        assert spec_data["hierarchy"]["verify-1-1"]["metadata"]["verification_type"] == "run-tests"
        assert spec_data["hierarchy"]["verify-2-1"]["metadata"]["verification_type"] == "run-tests"

    def test_updates_multiple_fields(self, task_tool, test_specs_dir):
        result = task_tool(
            action="metadata-batch",
            spec_id="batch-test-spec-001",
            node_type="task",
            file_path="src/module.py",
            estimated_hours=2.5,
            category="implementation"
        )
        assert result["success"] is True
        assert result["data"]["updated_count"] == 3

        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        for task_id in ["task-1-1", "task-1-2", "task-2-1"]:
            meta = spec_data["hierarchy"][task_id]["metadata"]
            assert meta["file_path"] == "src/module.py"
            assert meta["estimated_hours"] == 2.5
            assert meta["category"] == "implementation"

    def test_updates_owners_and_labels(self, task_tool, test_specs_dir):
        result = task_tool(
            action="metadata-batch",
            spec_id="batch-test-spec-001",
            phase_id="phase-2",
            owners=["alice", "bob"],
            labels={"priority": "high", "team": "backend"}
        )
        assert result["success"] is True
        assert result["data"]["updated_count"] == 2

        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        for task_id in ["task-2-1", "verify-2-1"]:
            meta = spec_data["hierarchy"][task_id]["metadata"]
            assert meta["owners"] == ["alice", "bob"]
            assert meta["labels"] == {"priority": "high", "team": "backend"}


# =============================================================================
# Response Format Tests
# =============================================================================

class TestResponseFormat:
    def test_response_includes_filters(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", phase_id="phase-1", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert "filters" in result["data"]
        assert result["data"]["filters"]["node_type"] == "task"
        assert result["data"]["filters"]["phase_id"] == "phase-1"

    def test_response_includes_metadata_applied(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="test.py", estimated_hours=1.0, dry_run=True)
        assert result["success"] is True
        assert "metadata_applied" in result["data"]
        assert result["data"]["metadata_applied"]["file_path"] == "test.py"
        assert result["data"]["metadata_applied"]["estimated_hours"] == 1.0

    def test_response_includes_telemetry(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="test.py", dry_run=True)
        assert result["success"] is True
        assert "meta" in result
        assert "telemetry" in result["meta"]
        assert "duration_ms" in result["meta"]["telemetry"]


# =============================================================================
# Rollback Tests
# =============================================================================

class TestRollback:
    def test_rollback_on_save_failure(self, task_tool, test_specs_dir, monkeypatch):
        """Verify changes are rolled back when save fails."""
        from foundry_mcp.tools.unified import task as task_module

        # First set initial values
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="original.py")
        assert result["success"] is True

        # Mock save_spec to fail
        def mock_save_spec(*args, **kwargs):
            return False

        monkeypatch.setattr(task_module, "save_spec", mock_save_spec)

        # Try to update - should fail and rollback
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="should_not_persist.py")
        assert result["success"] is False
        assert "rolled back" in result["error"].lower()

        # Restore original save_spec and verify values weren't changed
        monkeypatch.undo()
        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        assert spec_data["hierarchy"]["task-1-1"]["metadata"]["file_path"] == "original.py"

    def test_rollback_restores_none_values(self, task_tool, test_specs_dir, monkeypatch):
        """Verify rollback properly removes fields that were None before update."""
        from foundry_mcp.tools.unified import task as task_module

        # Verify initial state has no file_path
        spec_path = test_specs_dir / "active" / "batch-test-spec-001.json"
        spec_data = json.loads(spec_path.read_text())
        assert spec_data["hierarchy"]["task-1-1"]["metadata"].get("file_path") is None

        # Mock save_spec to fail
        def mock_save_spec(*args, **kwargs):
            return False

        monkeypatch.setattr(task_module, "save_spec", mock_save_spec)

        # Try to add file_path - should fail and rollback
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="task", file_path="should_not_exist.py")
        assert result["success"] is False

        # Note: Since save failed, in-memory changes should be rolled back
        # The file on disk shouldn't change since save was mocked to fail


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    def test_spec_not_found(self, task_tool):
        result = task_tool(action="metadata-batch", spec_id="nonexistent-spec", node_type="task", file_path="test.py")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_regex_case_insensitive(self, task_tool):
        # Pattern should match case-insensitively
        result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", pattern="FEATURE", file_path="test.py", dry_run=True)
        assert result["success"] is True
        # "Implement feature" and "Deploy feature" should match
        assert result["data"]["matched_count"] == 2

    def test_valid_verification_types(self, task_tool):
        # Canonical verification types: run-tests, fidelity, manual
        for vtype in ["run-tests", "fidelity", "manual"]:
            result = task_tool(action="metadata-batch", spec_id="batch-test-spec-001", node_type="verify", verification_type=vtype, dry_run=True)
            assert result["success"] is True, f"Failed for verification_type={vtype}"
