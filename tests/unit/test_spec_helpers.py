"""
Unit tests for spec helper tools.

Tests the spec-find-related-files, spec-find-patterns, spec-detect-cycles,
and spec-validate-paths tools. These tools use direct Python API calls
instead of CLI subprocess calls.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance."""
    mcp = MagicMock()
    mcp._tools = {}

    def mock_tool(*args, **kwargs):
        def decorator(func):
            tool_name = kwargs.get("name", func.__name__)
            mcp._tools[tool_name] = func
            return func
        return decorator

    mcp.tool = mock_tool
    return mcp


@pytest.fixture
def mock_config():
    """Create a mock server config."""
    config = MagicMock()
    config.project_root = "/test/project"
    config.specs_dir = None
    return config


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project with specs and source files."""
    # Create specs directory structure
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()

    # Create a sample spec
    spec_data = {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "version": "1.0.0",
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "pending",
                "children": ["phase-1"],
                "parent": None,
                "metadata": {}
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "pending",
                "children": ["task-1-1", "task-1-2"],
                "parent": "spec-root",
                "metadata": {}
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
                "metadata": {
                    "file_path": "src/main.py"
                },
                "dependencies": {}
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 2",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
                "metadata": {
                    "file_path": "src/helper.py"
                },
                "dependencies": {
                    "blocked_by": ["task-1-1"]
                }
            }
        }
    }

    spec_file = specs_dir / "active" / "test-spec-001.json"
    with open(spec_file, "w") as f:
        json.dump(spec_data, f)

    # Create some source files
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("# main module\n")
    (src_dir / "helper.py").write_text("# helper module\n")

    return tmp_path, spec_data


# =============================================================================
# spec-find-related-files Tool Tests
# =============================================================================


class TestSpecFindRelatedFiles:
    """Tests for spec-find-related-files tool."""

    def test_validation_error_on_empty_file_path(self, mock_mcp, mock_config, assert_response_contract):
        """Tool should return validation error on empty file_path."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        register_spec_helper_tools(mock_mcp, mock_config)

        find_related = mock_mcp._tools["spec-find-related-files"]
        result = find_related(file_path="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"
        assert result["data"].get("error_type") == "validation"

    def test_returns_related_files(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Tool should return related files found in specs."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        # Change to the project directory so find_specs_directory works
        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        find_related = mock_mcp._tools["spec-find-related-files"]
        result = find_related(file_path="src/main.py")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["file_path"] == "src/main.py"
        assert "related_files" in result["data"]
        assert "total_count" in result["data"]

    def test_returns_empty_when_no_specs_dir(self, mock_mcp, mock_config, assert_response_contract, tmp_path, monkeypatch):
        """Tool should return error when specs directory not found."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Use a directory without specs
        monkeypatch.chdir(tmp_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        find_related = mock_mcp._tools["spec-find-related-files"]
        result = find_related(file_path="some/file.py")

        assert_response_contract(result)
        assert result["success"] is False


# =============================================================================
# spec-find-patterns Tool Tests
# =============================================================================


class TestSpecFindPatterns:
    """Tests for spec-find-patterns tool."""

    def test_validation_error_on_empty_pattern(self, mock_mcp, mock_config, assert_response_contract):
        """Tool should return validation error on empty pattern."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        register_spec_helper_tools(mock_mcp, mock_config)

        find_patterns = mock_mcp._tools["spec-find-patterns"]
        result = find_patterns(pattern="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"
        assert result["data"].get("error_type") == "validation"

    def test_finds_matching_files(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Tool should find files matching the pattern."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        find_patterns = mock_mcp._tools["spec-find-patterns"]
        result = find_patterns(pattern="src/*.py")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["pattern"] == "src/*.py"
        assert "matches" in result["data"]
        assert "total_count" in result["data"]
        # Should find main.py and helper.py
        assert len(result["data"]["matches"]) == 2

    def test_finds_no_matches(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Tool should return empty matches for non-matching pattern."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        find_patterns = mock_mcp._tools["spec-find-patterns"]
        result = find_patterns(pattern="*.nonexistent")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["matches"] == []
        assert result["data"]["total_count"] == 0

    def test_with_directory_scope(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Tool should scope search to specified directory."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        find_patterns = mock_mcp._tools["spec-find-patterns"]
        result = find_patterns(pattern="*.py", directory=str(project_path / "src"))

        assert_response_contract(result)
        assert result["success"] is True
        assert "directory" in result["data"]

    def test_invalid_directory(self, mock_mcp, mock_config, assert_response_contract):
        """Tool should return error for non-existent directory."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        register_spec_helper_tools(mock_mcp, mock_config)

        find_patterns = mock_mcp._tools["spec-find-patterns"]
        result = find_patterns(pattern="*.py", directory="/nonexistent/path")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_FOUND"


# =============================================================================
# spec-detect-cycles Tool Tests
# =============================================================================


class TestSpecDetectCycles:
    """Tests for spec-detect-cycles tool."""

    def test_validation_error_on_empty_spec_id(self, mock_mcp, mock_config, assert_response_contract):
        """Tool should return validation error on empty spec_id."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        register_spec_helper_tools(mock_mcp, mock_config)

        detect_cycles = mock_mcp._tools["spec-detect-cycles"]
        result = detect_cycles(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"
        assert result["data"].get("error_type") == "validation"

    def test_no_cycles_detected(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Tool should detect no cycles in acyclic spec."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        detect_cycles = mock_mcp._tools["spec-detect-cycles"]
        result = detect_cycles(spec_id="test-spec-001")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["has_cycles"] is False
        assert result["data"]["cycles"] == []
        assert result["data"]["cycle_count"] == 0

    def test_spec_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Tool should return error for non-existent spec."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        detect_cycles = mock_mcp._tools["spec-detect-cycles"]
        result = detect_cycles(spec_id="nonexistent-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "SPEC_NOT_FOUND"

    def test_detects_cycles(self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch):
        """Tool should detect cycles in cyclic spec."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        # Create a spec with a dependency cycle
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        cyclic_spec = {
            "spec_id": "cyclic-spec",
            "title": "Cyclic Specification",
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Cyclic Spec",
                    "children": ["phase-1"],
                    "parent": None,
                    "dependencies": {}
                },
                "phase-1": {
                    "type": "phase",
                    "title": "Phase 1",
                    "children": ["task-a", "task-b"],
                    "parent": "spec-root",
                    "dependencies": {}
                },
                "task-a": {
                    "type": "task",
                    "title": "Task A",
                    "children": [],
                    "parent": "phase-1",
                    "dependencies": {
                        "blocked_by": ["task-b"]  # A depends on B
                    }
                },
                "task-b": {
                    "type": "task",
                    "title": "Task B",
                    "children": [],
                    "parent": "phase-1",
                    "dependencies": {
                        "blocked_by": ["task-a"]  # B depends on A -> cycle!
                    }
                }
            }
        }

        spec_file = specs_dir / "cyclic-spec.json"
        with open(spec_file, "w") as f:
            json.dump(cyclic_spec, f)

        monkeypatch.chdir(tmp_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        detect_cycles = mock_mcp._tools["spec-detect-cycles"]
        result = detect_cycles(spec_id="cyclic-spec")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["has_cycles"] is True
        assert result["data"]["cycle_count"] > 0
        assert len(result["data"]["affected_tasks"]) > 0


# =============================================================================
# spec-validate-paths Tool Tests
# =============================================================================


class TestSpecValidatePaths:
    """Tests for spec-validate-paths tool."""

    def test_validation_error_on_empty_paths(self, mock_mcp, mock_config, assert_response_contract):
        """Tool should return validation error on empty paths."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        register_spec_helper_tools(mock_mcp, mock_config)

        validate_paths = mock_mcp._tools["spec-validate-paths"]
        result = validate_paths(paths=[])

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"
        assert result["data"].get("error_type") == "validation"

    def test_validates_existing_paths(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Tool should validate paths that exist."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        validate_paths = mock_mcp._tools["spec-validate-paths"]
        result = validate_paths(paths=["src/main.py", "src/helper.py"])

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["paths_checked"] == 2
        assert result["data"]["valid_count"] == 2
        assert result["data"]["invalid_count"] == 0
        assert result["data"]["all_valid"] is True

    def test_validates_nonexistent_paths(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Tool should identify paths that don't exist."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        validate_paths = mock_mcp._tools["spec-validate-paths"]
        result = validate_paths(paths=["src/main.py", "nonexistent.py"])

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["paths_checked"] == 2
        assert result["data"]["valid_count"] == 1
        assert result["data"]["invalid_count"] == 1
        assert result["data"]["all_valid"] is False
        assert "src/main.py" in result["data"]["valid_paths"]
        assert "nonexistent.py" in result["data"]["invalid_paths"]

    def test_with_base_directory(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Tool should resolve paths relative to base_directory."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        validate_paths = mock_mcp._tools["spec-validate-paths"]
        result = validate_paths(
            paths=["main.py", "helper.py"],
            base_directory=str(project_path / "src")
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["all_valid"] is True
        assert "base_directory" in result["data"]


# =============================================================================
# Integration Tests for Tool Registration
# =============================================================================


class TestToolRegistration:
    """Test that all tools are properly registered."""

    def test_all_tools_registered(self, mock_mcp, mock_config):
        """All spec helper tools should be registered with the MCP server."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        register_spec_helper_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec-find-related-files",
            "spec-find-patterns",
            "spec-detect-cycles",
            "spec-validate-paths",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        register_spec_helper_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_validation_error_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """Validation error responses should have correct structure."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        register_spec_helper_tools(mock_mcp, mock_config)

        find_patterns = mock_mcp._tools["spec-find-patterns"]
        result = find_patterns(pattern="")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_success_response_has_required_fields(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Success responses should have all required fields."""
        from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

        project_path, _ = temp_project

        monkeypatch.chdir(project_path)

        register_spec_helper_tools(mock_mcp, mock_config)

        detect_cycles = mock_mcp._tools["spec-detect-cycles"]
        result = detect_cycles(spec_id="test-spec-001")

        # Validate structure
        assert result["success"] is True
        assert result["error"] is None
        assert result["meta"]["version"] == "response-v2"
