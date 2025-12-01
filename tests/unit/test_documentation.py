"""
Unit tests for foundry_mcp.tools.documentation module.

Tests cover:
- spec-doc tool for generating human-readable documentation
- spec-doc-llm tool (returns NOT_IMPLEMENTED since it requires LLM integration)
- spec-review-fidelity tool (returns NOT_IMPLEMENTED since it requires AI consultation)
- Response contract compliance for all tools
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
    """Create a mock ServerConfig."""
    config = MagicMock()
    config.project_root = "/test/project"
    config.specs_dir = None
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
    (specs_dir / ".human-readable").mkdir()

    # Create a sample spec
    spec_data = {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "version": "1.0.0",
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "in_progress",
                "children": ["phase-1"],
                "parent": None,
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "children": ["task-1-1", "task-1-2"],
                "parent": "spec-root",
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "completed",
                "children": [],
                "parent": "phase-1",
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 2",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
            },
        },
    }

    spec_file = specs_dir / "active" / "test-spec-001.json"
    with open(spec_file, "w") as f:
        json.dump(spec_data, f)

    return tmp_path, spec_data


# =============================================================================
# spec-doc Tool Tests
# =============================================================================


class TestSpecDoc:
    """Tests for spec-doc tool."""

    def test_invalid_output_format_returns_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return validation error for invalid output_format."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec", output_format="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"
        assert result["data"].get("error_type") == "validation"

    def test_invalid_mode_returns_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return validation error for invalid mode."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec", mode="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_specs_dir_not_found(self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch):
        """Should handle missing specs directory."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Use an empty directory without specs
        monkeypatch.chdir(tmp_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "SPECS_DIR_NOT_FOUND"

    def test_spec_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle spec not found."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="nonexistent-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "SPEC_NOT_FOUND"

    def test_successful_documentation_generation(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should generate documentation successfully."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec-001")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["format"] == "markdown"
        assert "output_path" in result["data"]
        assert "stats" in result["data"]

    def test_custom_output_path(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should use custom output path when provided."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        custom_path = str(project_path / "custom_output.md")
        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec-001", output_path=custom_path)

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["output_path"] == custom_path


# =============================================================================
# spec-doc-llm Tool Tests
# =============================================================================


class TestSpecDocLlm:
    """Tests for spec-doc-llm tool."""

    def test_returns_not_implemented(self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch):
        """Should return NOT_IMPLEMENTED since it requires LLM integration."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        monkeypatch.chdir(tmp_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory=str(tmp_path))

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_IMPLEMENTED"
        assert result["data"].get("error_type") == "unavailable"

    def test_includes_remediation(self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch):
        """Should include remediation guidance."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        monkeypatch.chdir(tmp_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory=str(tmp_path))

        assert_response_contract(result)
        assert "remediation" in result["data"]
        assert "sdd-toolkit:llm-doc-gen" in result["data"]["remediation"]

    def test_missing_directory_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing directory."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_invalid_batch_size_validation(self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch):
        """Should return error for invalid batch_size."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        monkeypatch.chdir(tmp_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory=str(tmp_path), batch_size=20)

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_directory_not_found(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle directory not found."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory="/nonexistent/path")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_FOUND"


# =============================================================================
# spec-review-fidelity Tool Tests
# =============================================================================


class TestSpecReviewFidelity:
    """Tests for spec-review-fidelity tool."""

    def test_returns_not_implemented(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should return NOT_IMPLEMENTED since it requires AI consultation."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="test-spec-001")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_IMPLEMENTED"
        assert result["data"].get("error_type") == "unavailable"

    def test_includes_remediation(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should include remediation guidance."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="test-spec-001")

        assert_response_contract(result)
        assert "remediation" in result["data"]
        assert "sdd-toolkit:sdd-fidelity-review" in result["data"]["remediation"]

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_mutual_exclusivity_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when both task_id and phase_id are provided."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(
            spec_id="test-spec",
            task_id="task-1-1",
            phase_id="phase-1",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_invalid_consensus_threshold_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid consensus_threshold."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(
            spec_id="test-spec",
            consensus_threshold=10,
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_spec_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle spec not found."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="nonexistent-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "SPEC_NOT_FOUND"


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestDocumentationToolRegistration:
    """Tests for documentation tool registration."""

    def test_all_tools_registered(self, mock_mcp, mock_config):
        """All documentation tools should be registered."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec-doc",
            "spec-doc-llm",
            "spec-review-fidelity",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_validation_error_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """Validation error responses should have correct structure."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec", mode="invalid")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_success_response_has_required_fields(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Success responses should have all required fields."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec-001")

        # Validate structure
        assert result["success"] is True
        assert result["error"] is None
        assert result["meta"]["version"] == "response-v2"

    def test_not_implemented_response_structure(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """NOT_IMPLEMENTED responses should have correct structure."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="test-spec-001")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"
        assert result["success"] is False
        assert result["data"]["error_code"] == "NOT_IMPLEMENTED"


# =============================================================================
# Telemetry and Metrics Tests
# =============================================================================


class TestTelemetryMetrics:
    """Tests for telemetry and metrics integration."""

    def test_metrics_collector_accessible(self):
        """Metrics collector should be accessible."""
        from foundry_mcp.core.observability import get_metrics

        metrics = get_metrics()
        assert metrics is not None
        assert hasattr(metrics, "counter")
        assert hasattr(metrics, "timer")

    def test_metrics_collector_accepts_labels(self):
        """Metrics collector should accept labels."""
        from foundry_mcp.core.observability import MetricsCollector

        metrics = MetricsCollector()

        # Counter should accept labels
        metrics.counter("documentation.success", labels={"tool": "spec-doc"})
        metrics.counter("documentation.errors", labels={"tool": "spec-doc", "error_type": "not_found"})

        # Timer should also accept labels
        metrics.timer("documentation.spec_doc_time", 100.5, labels={"mode": "basic"})


# =============================================================================
# Response Envelope Compliance Tests
# =============================================================================


class TestResponseEnvelopeCompliance:
    """Tests for response envelope compliance."""

    def test_success_response_has_required_fields(self):
        """Test success response includes required envelope fields."""
        from foundry_mcp.core.responses import success_response
        from dataclasses import asdict

        response = asdict(success_response(
            spec_id="test-spec",
            format="markdown",
            output_path="test.md",
        ))

        assert "success" in response
        assert response["success"] is True
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(self):
        """Test error response includes required envelope fields."""
        from foundry_mcp.core.responses import error_response
        from dataclasses import asdict

        response = asdict(error_response(
            "Test error message",
            error_code="VALIDATION_ERROR",
            error_type="validation",
        ))

        assert "success" in response
        assert response["success"] is False
        assert "error" in response
        assert response["error"] == "Test error message"
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"
