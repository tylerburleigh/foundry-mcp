"""
Unit tests for foundry_mcp.tools.analysis module.

Tests the analysis tools for SDD specifications, including validation,
direct core API integration, and response contracts.
"""

import json
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
    return config


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project with specs."""
    # Create specs directory structure
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()

    # Create a sample spec with dependencies
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
                "parent": None
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "pending",
                "children": ["task-1-1", "task-1-2", "task-1-3"],
                "parent": "spec-root"
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
                "dependencies": {}
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 2",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
                "dependencies": {
                    "blocked_by": ["task-1-1"]
                }
            },
            "task-1-3": {
                "type": "task",
                "title": "Task 3",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
                "dependencies": {
                    "blocked_by": ["task-1-1"]
                }
            }
        }
    }

    spec_file = specs_dir / "active" / "test-spec-001.json"
    with open(spec_file, "w") as f:
        json.dump(spec_data, f)

    return tmp_path, spec_data


# =============================================================================
# spec-analyze Tool Tests
# =============================================================================


class TestSpecAnalyze:
    """Test the spec-analyze tool."""

    def test_basic_analysis(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should analyze specs successfully."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_analysis_tools(mock_mcp, mock_config)

        spec_analyze = mock_mcp._tools["spec-analyze"]
        result = spec_analyze()

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["has_specs"] is True
        assert "spec_counts" in result["data"]
        assert "total_specs" in result["data"]

    def test_analysis_with_directory(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should pass directory parameter."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_analysis_tools(mock_mcp, mock_config)

        spec_analyze = mock_mcp._tools["spec-analyze"]
        result = spec_analyze(directory=str(project_path))

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["directory"] == str(project_path)

    def test_no_specs_directory(self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch):
        """Should handle missing specs directory."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        # Empty directory without specs
        monkeypatch.chdir(tmp_path)

        register_analysis_tools(mock_mcp, mock_config)

        spec_analyze = mock_mcp._tools["spec-analyze"]
        result = spec_analyze()

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["has_specs"] is False


# =============================================================================
# review-parse-feedback Tool Tests
# =============================================================================


class TestReviewParseFeedback:
    """Test the review-parse-feedback tool."""

    def test_returns_not_implemented(self, mock_mcp, mock_config, assert_response_contract):
        """Should return NOT_IMPLEMENTED since it requires complex parsing."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        register_analysis_tools(mock_mcp, mock_config)

        parse_feedback = mock_mcp._tools["review-parse-feedback"]
        result = parse_feedback(spec_id="test-spec", review_path="/path/to/review.md")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_IMPLEMENTED"
        assert result["data"].get("error_type") == "unavailable"

    def test_includes_remediation(self, mock_mcp, mock_config, assert_response_contract):
        """Should include remediation guidance."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        register_analysis_tools(mock_mcp, mock_config)

        parse_feedback = mock_mcp._tools["review-parse-feedback"]
        result = parse_feedback(spec_id="test-spec", review_path="/path/to/review.md")

        assert_response_contract(result)
        assert "remediation" in result["data"]
        assert "sdd-toolkit:sdd-modify" in result["data"]["remediation"]

    def test_returns_request_parameters_in_data(self, mock_mcp, mock_config, assert_response_contract):
        """Should return request parameters in response data."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        register_analysis_tools(mock_mcp, mock_config)

        parse_feedback = mock_mcp._tools["review-parse-feedback"]
        result = parse_feedback(
            spec_id="test-spec-001",
            review_path="/path/to/review.md",
            output_path="/path/to/output.json"
        )

        assert_response_contract(result)
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["review_path"] == "/path/to/review.md"
        assert result["data"]["output_path"] == "/path/to/output.json"


# =============================================================================
# spec-analyze-deps Tool Tests
# =============================================================================


class TestSpecAnalyzeDeps:
    """Test the spec-analyze-deps tool."""

    def test_validation_error_on_empty_spec_id(self, mock_mcp, mock_config, assert_response_contract):
        """Should return validation error on empty spec_id."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        register_analysis_tools(mock_mcp, mock_config)

        analyze_deps = mock_mcp._tools["spec-analyze-deps"]
        result = analyze_deps(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"
        assert result["data"].get("error_type") == "validation"

    def test_basic_dependency_analysis(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should analyze dependencies successfully."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_analysis_tools(mock_mcp, mock_config)

        analyze_deps = mock_mcp._tools["spec-analyze-deps"]
        result = analyze_deps(spec_id="test-spec-001")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert "dependency_count" in result["data"]
        assert "bottlenecks" in result["data"]
        assert "circular_deps" in result["data"]

    def test_spec_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle spec not found errors."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_analysis_tools(mock_mcp, mock_config)

        analyze_deps = mock_mcp._tools["spec-analyze-deps"]
        result = analyze_deps(spec_id="nonexistent-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_FOUND"

    def test_bottleneck_threshold(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should use bottleneck_threshold parameter."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_analysis_tools(mock_mcp, mock_config)

        analyze_deps = mock_mcp._tools["spec-analyze-deps"]
        result = analyze_deps(spec_id="test-spec-001", bottleneck_threshold=1)

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["bottleneck_threshold"] == 1
        # task-1-1 blocks 2 tasks, so with threshold 1 it should be a bottleneck
        assert len(result["data"]["bottlenecks"]) > 0


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestAnalysisToolRegistration:
    """Test that all analysis tools are registered correctly."""

    def test_all_tools_registered(self, mock_mcp, mock_config):
        """All analysis tools should be registered with the MCP server."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        register_analysis_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec-analyze",
            "review-parse-feedback",
            "spec-analyze-deps",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        register_analysis_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_validation_error_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """Validation error responses should have correct structure."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        register_analysis_tools(mock_mcp, mock_config)

        analyze_deps = mock_mcp._tools["spec-analyze-deps"]
        result = analyze_deps(spec_id="")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_success_response_has_required_fields(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Success responses should have all required fields."""
        from foundry_mcp.tools.analysis import register_analysis_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_analysis_tools(mock_mcp, mock_config)

        spec_analyze = mock_mcp._tools["spec-analyze"]
        result = spec_analyze()

        # Validate structure
        assert result["success"] is True
        assert result["error"] is None
        assert result["meta"]["version"] == "response-v2"
