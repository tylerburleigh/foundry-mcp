"""
Unit tests for foundry_mcp.tools.planning module.

Tests the planning tools for SDD specifications, including:
- plan-format - Format task plans for sharing
- phase-list - Enumerate phases in a spec
- phase-check-complete - Verify completion readiness
- phase-report-time - Time tracking per phase
- spec-reconcile-state - Returns NOT_IMPLEMENTED (requires git integration)
- plan-report-time - Aggregate time tracking reports
- spec-audit - Returns NOT_IMPLEMENTED (requires AI analysis)

Uses direct Python core API calls, not CLI subprocess calls.
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

    # Create a sample spec with multiple phases and tasks
    spec_data = {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "version": "1.0.0",
        "metadata": {
            "title": "Test Specification"
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "in_progress",
                "children": ["phase-1", "phase-2"],
                "parent": None,
            },
            "phase-1": {
                "type": "phase",
                "title": "Setup Phase",
                "status": "completed",
                "children": ["task-1-1", "task-1-2"],
                "parent": "spec-root",
            },
            "task-1-1": {
                "type": "task",
                "title": "Initial Setup",
                "status": "completed",
                "children": [],
                "parent": "phase-1",
                "metadata": {
                    "estimated_hours": 2.0,
                    "actual_hours": 2.5,
                },
            },
            "task-1-2": {
                "type": "task",
                "title": "Configuration",
                "status": "completed",
                "children": [],
                "parent": "phase-1",
                "metadata": {
                    "estimated_hours": 1.0,
                    "actual_hours": 1.0,
                },
            },
            "phase-2": {
                "type": "phase",
                "title": "Implementation Phase",
                "status": "in_progress",
                "children": ["task-2-1", "task-2-2"],
                "parent": "spec-root",
            },
            "task-2-1": {
                "type": "task",
                "title": "Core Implementation",
                "status": "completed",
                "children": [],
                "parent": "phase-2",
                "metadata": {
                    "estimated_hours": 4.0,
                    "actual_hours": 5.0,
                },
            },
            "task-2-2": {
                "type": "task",
                "title": "Integration",
                "status": "pending",
                "children": [],
                "parent": "phase-2",
                "metadata": {
                    "estimated_hours": 3.0,
                },
            },
        },
    }

    spec_file = specs_dir / "active" / "test-spec-001.json"
    with open(spec_file, "w") as f:
        json.dump(spec_data, f)

    return tmp_path, spec_data


# =============================================================================
# plan-format Tool Tests
# =============================================================================


class TestPlanFormat:
    """Test the plan-format tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        plan_format = mock_mcp._tools["plan-format"]
        result = plan_format(spec_id="", task_id="task-1-1")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_missing_task_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing task_id."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        plan_format = mock_mcp._tools["plan-format"]
        result = plan_format(spec_id="test-spec", task_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_basic_plan_formatting(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should format plan successfully."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        plan_format = mock_mcp._tools["plan-format"]
        result = plan_format(spec_id="test-spec-001", task_id="task-1-1")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["task_id"] == "task-1-1"
        assert "formatted" in result["data"]

    def test_task_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle task not found."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        plan_format = mock_mcp._tools["plan-format"]
        result = plan_format(spec_id="test-spec-001", task_id="nonexistent-task")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_FOUND"

    def test_spec_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle spec not found."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        plan_format = mock_mcp._tools["plan-format"]
        result = plan_format(spec_id="nonexistent-spec", task_id="task-1-1")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_FOUND"


# =============================================================================
# phase-list Tool Tests
# =============================================================================


class TestPhaseList:
    """Test the phase-list tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        phase_list = mock_mcp._tools["phase-list"]
        result = phase_list(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_basic_phase_listing(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should list phases successfully."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        phase_list = mock_mcp._tools["phase-list"]
        result = phase_list(spec_id="test-spec-001")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["total_phases"] == 2
        assert result["data"]["completed_phases"] == 1
        assert len(result["data"]["phases"]) == 2

    def test_spec_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle spec not found."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        phase_list = mock_mcp._tools["phase-list"]
        result = phase_list(spec_id="nonexistent-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_FOUND"


# =============================================================================
# phase-check-complete Tool Tests
# =============================================================================


class TestPhaseCheckComplete:
    """Test the phase-check-complete tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        phase_check_complete = mock_mcp._tools["phase-check-complete"]
        result = phase_check_complete(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_mutual_exclusivity_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when both phase_id and task_id provided."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        phase_check_complete = mock_mcp._tools["phase-check-complete"]
        result = phase_check_complete(
            spec_id="test-spec",
            phase_id="phase-1",
            task_id="task-1-1",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "INVALID_PARAMS"

    def test_phase_complete(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should verify completed phase."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        phase_check_complete = mock_mcp._tools["phase-check-complete"]
        result = phase_check_complete(spec_id="test-spec-001", phase_id="phase-1")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["scope"] == "phase"
        assert result["data"]["phase_id"] == "phase-1"
        assert result["data"]["is_complete"] is True

    def test_phase_incomplete(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should detect incomplete phase."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        phase_check_complete = mock_mcp._tools["phase-check-complete"]
        result = phase_check_complete(spec_id="test-spec-001", phase_id="phase-2")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["scope"] == "phase"
        assert result["data"]["is_complete"] is False
        assert len(result["data"]["pending_tasks"]) > 0


# =============================================================================
# phase-report-time Tool Tests
# =============================================================================


class TestPhaseReportTime:
    """Test the phase-report-time tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        phase_report_time = mock_mcp._tools["phase-report-time"]
        result = phase_report_time(spec_id="", phase_id="phase-1")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_missing_phase_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing phase_id."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        phase_report_time = mock_mcp._tools["phase-report-time"]
        result = phase_report_time(spec_id="test-spec", phase_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_basic_time_report(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should report phase time successfully."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        phase_report_time = mock_mcp._tools["phase-report-time"]
        result = phase_report_time(spec_id="test-spec-001", phase_id="phase-1")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["phase_id"] == "phase-1"
        assert result["data"]["estimated_hours"] == 3.0  # 2 + 1
        assert result["data"]["actual_hours"] == 3.5  # 2.5 + 1.0
        assert result["data"]["variance_hours"] == 0.5
        assert result["data"]["task_count"] == 2
        assert result["data"]["completed_count"] == 2

    def test_phase_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle phase not found."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        phase_report_time = mock_mcp._tools["phase-report-time"]
        result = phase_report_time(spec_id="test-spec-001", phase_id="nonexistent-phase")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_FOUND"


# =============================================================================
# spec-reconcile-state Tool Tests
# =============================================================================


class TestSpecReconcileState:
    """Test the spec-reconcile-state tool."""

    def test_returns_not_implemented(self, mock_mcp, mock_config, assert_response_contract):
        """Should return NOT_IMPLEMENTED since it requires git integration."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        spec_reconcile_state = mock_mcp._tools["spec-reconcile-state"]
        result = spec_reconcile_state(spec_id="test-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_IMPLEMENTED"
        assert result["data"].get("error_type") == "unavailable"

    def test_includes_remediation(self, mock_mcp, mock_config, assert_response_contract):
        """Should include remediation guidance."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        spec_reconcile_state = mock_mcp._tools["spec-reconcile-state"]
        result = spec_reconcile_state(spec_id="test-spec")

        assert_response_contract(result)
        assert "remediation" in result["data"]
        assert "sdd-toolkit:sdd-fidelity-review" in result["data"]["remediation"]


# =============================================================================
# plan-report-time Tool Tests
# =============================================================================


class TestPlanReportTime:
    """Test the plan-report-time tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        plan_report_time = mock_mcp._tools["plan-report-time"]
        result = plan_report_time(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_basic_aggregate_report(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should generate aggregate time report."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        plan_report_time = mock_mcp._tools["plan-report-time"]
        result = plan_report_time(spec_id="test-spec-001")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["total_estimated_hours"] == 10.0  # 2 + 1 + 4 + 3
        assert result["data"]["total_actual_hours"] == 8.5  # 2.5 + 1.0 + 5.0 + 0
        assert len(result["data"]["phases"]) == 2
        assert result["data"]["total_tasks"] == 4
        assert result["data"]["completed_tasks"] == 3

    def test_spec_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle spec not found."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        plan_report_time = mock_mcp._tools["plan-report-time"]
        result = plan_report_time(spec_id="nonexistent-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_FOUND"


# =============================================================================
# spec-audit Tool Tests
# =============================================================================


class TestSpecAudit:
    """Test the spec-audit tool."""

    def test_returns_not_implemented(self, mock_mcp, mock_config, assert_response_contract):
        """Should return NOT_IMPLEMENTED since it requires AI analysis."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        spec_audit = mock_mcp._tools["spec-audit"]
        result = spec_audit(spec_id="test-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_IMPLEMENTED"
        assert result["data"].get("error_type") == "unavailable"

    def test_includes_remediation(self, mock_mcp, mock_config, assert_response_contract):
        """Should include remediation guidance."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        spec_audit = mock_mcp._tools["spec-audit"]
        result = spec_audit(spec_id="test-spec")

        assert_response_contract(result)
        assert "remediation" in result["data"]
        assert "sdd-toolkit:sdd-plan-review" in result["data"]["remediation"]


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestPlanningToolRegistration:
    """Test that all planning tools are properly registered."""

    def test_all_planning_tools_registered(self, mock_mcp, mock_config):
        """All planning tools should be registered with the MCP server."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        expected_tools = [
            "plan-format",
            "phase-list",
            "phase-check-complete",
            "phase-report-time",
            "spec-reconcile-state",
            "plan-report-time",
            "spec-audit",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_validation_error_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """Validation error responses should have correct structure."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        phase_list = mock_mcp._tools["phase-list"]
        result = phase_list(spec_id="")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_success_response_has_required_fields(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Success responses should have all required fields."""
        from foundry_mcp.tools.planning import register_planning_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_planning_tools(mock_mcp, mock_config)

        phase_list = mock_mcp._tools["phase-list"]
        result = phase_list(spec_id="test-spec-001")

        # Validate structure
        assert result["success"] is True
        assert result["error"] is None
        assert result["meta"]["version"] == "response-v2"

    def test_not_implemented_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """NOT_IMPLEMENTED responses should have correct structure."""
        from foundry_mcp.tools.planning import register_planning_tools

        register_planning_tools(mock_mcp, mock_config)

        spec_audit = mock_mcp._tools["spec-audit"]
        result = spec_audit(spec_id="test-spec")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"
        assert result["success"] is False
        assert result["data"]["error_code"] == "NOT_IMPLEMENTED"
