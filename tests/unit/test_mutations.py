"""
Unit tests for foundry_mcp.tools.mutations and git_integration modules.

Tests the mutation and git integration tools for SDD specifications,
including validation, direct core API integration, and response contracts.
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
    """Create a temporary project with specs and modifications file."""
    # Create specs directory structure
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()

    # Create a sample spec with verification tasks
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
                "task_count": 3,
                "completed_count": 1,
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "children": ["task-1-1", "task-1-2", "verify-1-1"],
                "parent": "spec-root",
                "task_count": 3,
                "completed_count": 1,
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "completed",
                "children": [],
                "parent": "phase-1",
                "metadata": {
                    "estimated_hours": 2.0,
                    "complexity": "low",
                },
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 2",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
                "metadata": {},
            },
            "verify-1-1": {
                "type": "verify",
                "title": "Verification 1",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
                "metadata": {
                    "command": "pytest tests/",
                },
            },
        },
    }

    spec_file = specs_dir / "active" / "test-spec-001.json"
    with open(spec_file, "w") as f:
        json.dump(spec_data, f)

    # Create a modifications file
    modifications_data = {
        "modifications": [
            {
                "action": "update",
                "target": "task-1-2",
                "changes": {"status": "in_progress"},
            }
        ]
    }

    mods_file = tmp_path / "modifications.json"
    with open(mods_file, "w") as f:
        json.dump(modifications_data, f)

    return tmp_path, spec_data


# =============================================================================
# spec-apply-plan Tool Tests
# =============================================================================


class TestSpecApplyPlan:
    """Test the spec-apply-plan tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        spec_apply_plan = mock_mcp._tools["spec-apply-plan"]
        result = spec_apply_plan(spec_id="", modifications_file="/path/to/file.json")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"
        assert result["data"].get("error_type") == "validation"

    def test_missing_modifications_file_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing modifications_file."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        spec_apply_plan = mock_mcp._tools["spec-apply-plan"]
        result = spec_apply_plan(spec_id="test-spec", modifications_file="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_modifications_file_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle missing modifications file."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_mutation_tools(mock_mcp, mock_config)

        spec_apply_plan = mock_mcp._tools["spec-apply-plan"]
        result = spec_apply_plan(
            spec_id="test-spec-001",
            modifications_file="/nonexistent/file.json",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "FILE_NOT_FOUND"


# =============================================================================
# verification-add Tool Tests
# =============================================================================


class TestVerificationAdd:
    """Test the verification-add tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_add = mock_mcp._tools["verification-add"]
        result = verification_add(
            spec_id="",
            verify_id="verify-1-1",
            result="PASSED",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_missing_verify_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing verify_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_add = mock_mcp._tools["verification-add"]
        result = verification_add(
            spec_id="test-spec",
            verify_id="",
            result="PASSED",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_invalid_result_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid result."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_add = mock_mcp._tools["verification-add"]
        result = verification_add(
            spec_id="test-spec",
            verify_id="verify-1-1",
            result="INVALID",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_dry_run_mode(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should support dry_run mode."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_mutation_tools(mock_mcp, mock_config)

        verification_add = mock_mcp._tools["verification-add"]
        result = verification_add(
            spec_id="test-spec-001",
            verify_id="verify-1-1",
            result="PASSED",
            dry_run=True,
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert result["data"]["result"] == "PASSED"


# =============================================================================
# verification-execute Tool Tests
# =============================================================================


class TestVerificationExecute:
    """Test the verification-execute tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_execute = mock_mcp._tools["verification-execute"]
        result = verification_execute(
            spec_id="",
            verify_id="verify-1-1",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_missing_verify_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing verify_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_execute = mock_mcp._tools["verification-execute"]
        result = verification_execute(
            spec_id="test-spec",
            verify_id="",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"


# =============================================================================
# verification-format-summary Tool Tests
# =============================================================================


class TestVerificationFormatSummary:
    """Test the verification-format-summary tool."""

    def test_missing_input_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when no input source provided."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_format_summary = mock_mcp._tools["verification-format-summary"]
        result = verification_format_summary()

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_both_inputs_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when both inputs provided."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_format_summary = mock_mcp._tools["verification-format-summary"]
        result = verification_format_summary(
            json_file="/path/to/results.json",
            json_input='{"verifications": []}',
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"]["error_code"] == "VALIDATION_ERROR"

    def test_valid_json_input(self, mock_mcp, mock_config, assert_response_contract):
        """Should process valid JSON input successfully."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_format_summary = mock_mcp._tools["verification-format-summary"]
        result = verification_format_summary(json_input='{"verifications": []}')

        assert_response_contract(result)
        assert result["success"] is True

    def test_invalid_json_input(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid JSON input."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_format_summary = mock_mcp._tools["verification-format-summary"]
        result = verification_format_summary(json_input='not valid json')

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"]["error_code"] == "VALIDATION_ERROR"


# =============================================================================
# task-update-estimate Tool Tests
# =============================================================================


class TestTaskUpdateEstimate:
    """Test the task-update-estimate tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        task_update_estimate = mock_mcp._tools["task-update-estimate"]
        result = task_update_estimate(
            spec_id="",
            task_id="task-1-1",
            hours=4.5,
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_missing_task_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing task_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        task_update_estimate = mock_mcp._tools["task-update-estimate"]
        result = task_update_estimate(
            spec_id="test-spec",
            task_id="",
            hours=4.5,
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_no_update_fields_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when no update fields provided."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        task_update_estimate = mock_mcp._tools["task-update-estimate"]
        result = task_update_estimate(
            spec_id="test-spec",
            task_id="task-1-1",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_invalid_complexity_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid complexity."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        task_update_estimate = mock_mcp._tools["task-update-estimate"]
        result = task_update_estimate(
            spec_id="test-spec",
            task_id="task-1-1",
            complexity="super-hard",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_dry_run_mode(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should support dry_run mode."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_mutation_tools(mock_mcp, mock_config)

        task_update_estimate = mock_mcp._tools["task-update-estimate"]
        result = task_update_estimate(
            spec_id="test-spec-001",
            task_id="task-1-1",
            hours=4.5,
            dry_run=True,
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True


# =============================================================================
# task-update-metadata Tool Tests
# =============================================================================


class TestTaskUpdateMetadata:
    """Test the task-update-metadata tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        task_update_metadata = mock_mcp._tools["task-update-metadata"]
        result = task_update_metadata(
            spec_id="",
            task_id="task-1-1",
            file_path="src/module.py",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_no_fields_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error when no metadata fields provided."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        task_update_metadata = mock_mcp._tools["task-update-metadata"]
        result = task_update_metadata(
            spec_id="test-spec",
            task_id="task-1-1",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_invalid_verification_type_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid verification_type."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        task_update_metadata = mock_mcp._tools["task-update-metadata"]
        result = task_update_metadata(
            spec_id="test-spec",
            task_id="task-1-1",
            verification_type="invalid",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_invalid_metadata_json_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid metadata_json."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        task_update_metadata = mock_mcp._tools["task-update-metadata"]
        result = task_update_metadata(
            spec_id="test-spec",
            task_id="task-1-1",
            metadata_json="not valid json",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_dry_run_mode(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should support dry_run mode."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_mutation_tools(mock_mcp, mock_config)

        task_update_metadata = mock_mcp._tools["task-update-metadata"]
        result = task_update_metadata(
            spec_id="test-spec-001",
            task_id="task-1-1",
            file_path="src/module.py",
            dry_run=True,
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert "file_path" in result["data"]["fields_updated"]


# =============================================================================
# spec-sync-metadata Tool Tests
# =============================================================================


class TestSpecSyncMetadata:
    """Test the spec-sync-metadata tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        spec_sync_metadata = mock_mcp._tools["spec-sync-metadata"]
        result = spec_sync_metadata(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_spec_not_found(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should handle spec not found errors."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_mutation_tools(mock_mcp, mock_config)

        spec_sync_metadata = mock_mcp._tools["spec-sync-metadata"]
        result = spec_sync_metadata(spec_id="nonexistent-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "SPEC_NOT_FOUND"

    def test_dry_run_mode(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should support dry_run mode."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_mutation_tools(mock_mcp, mock_config)

        spec_sync_metadata = mock_mcp._tools["spec-sync-metadata"]
        result = spec_sync_metadata(
            spec_id="test-spec-001",
            dry_run=True,
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert result["data"]["synced"] is True


# =============================================================================
# Git Integration Module Tests
# =============================================================================


class TestTaskCreateCommit:
    """Test the task-create-commit tool."""

    def test_returns_not_implemented(self, mock_mcp, mock_config, assert_response_contract):
        """task_create_commit should return NOT_IMPLEMENTED since it requires git CLI."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools

        register_git_integration_tools(mock_mcp, mock_config)

        task_create_commit = mock_mcp._tools["task-create-commit"]
        result = task_create_commit(
            spec_id="test-spec",
            task_id="task-1-1",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_IMPLEMENTED"
        assert result["data"].get("error_type") == "unavailable"

    def test_includes_remediation(self, mock_mcp, mock_config, assert_response_contract):
        """Should include remediation guidance."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools

        register_git_integration_tools(mock_mcp, mock_config)

        task_create_commit = mock_mcp._tools["task-create-commit"]
        result = task_create_commit(
            spec_id="test-spec",
            task_id="task-1-1",
        )

        assert_response_contract(result)
        assert "remediation" in result["data"]
        assert "git commit" in result["data"]["remediation"].lower()


# =============================================================================
# journal-bulk-add Tool Tests
# =============================================================================


class TestJournalBulkAdd:
    """Test the journal-bulk-add tool."""

    def test_missing_spec_id_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools

        register_git_integration_tools(mock_mcp, mock_config)

        journal_bulk_add = mock_mcp._tools["journal-bulk-add"]
        result = journal_bulk_add(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_invalid_template_validation(self, mock_mcp, mock_config, assert_response_contract):
        """Should return error for invalid template."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools

        register_git_integration_tools(mock_mcp, mock_config)

        journal_bulk_add = mock_mcp._tools["journal-bulk-add"]
        result = journal_bulk_add(
            spec_id="test-spec",
            template="invalid_template",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_dry_run_mode(self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch):
        """Should support dry_run mode."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_git_integration_tools(mock_mcp, mock_config)

        journal_bulk_add = mock_mcp._tools["journal-bulk-add"]
        result = journal_bulk_add(
            spec_id="test-spec-001",
            tasks="task-1-1",
            template="completion",
            dry_run=True,
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["dry_run"] is True
        assert result["data"]["template_used"] == "completion"


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestMutationToolRegistration:
    """Test that all mutation tools are properly registered."""

    def test_all_mutation_tools_registered(self, mock_mcp, mock_config):
        """All mutation tools should be registered with the MCP server."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec-apply-plan",
            "verification-add",
            "verification-execute",
            "verification-format-summary",
            "task-update-estimate",
            "task-update-metadata",
            "spec-sync-metadata",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


class TestGitIntegrationToolRegistration:
    """Test that all git integration tools are properly registered."""

    def test_all_git_tools_registered(self, mock_mcp, mock_config):
        """All git integration tools should be registered."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools

        register_git_integration_tools(mock_mcp, mock_config)

        expected_tools = [
            "task-create-commit",
            "journal-bulk-add",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools

        register_git_integration_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_validation_error_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """Validation error responses should have correct structure."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        task_update_estimate = mock_mcp._tools["task-update-estimate"]
        result = task_update_estimate(spec_id="", task_id="task-1-1", hours=4.5)

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_success_response_has_required_fields(self, mock_mcp, mock_config, assert_response_contract):
        """Success responses should have all required fields."""
        from foundry_mcp.tools.mutations import register_mutation_tools

        register_mutation_tools(mock_mcp, mock_config)

        verification_format_summary = mock_mcp._tools["verification-format-summary"]
        result = verification_format_summary(json_input='{"verifications": []}')

        # Validate structure
        assert result["success"] is True
        assert result["error"] is None
        assert result["meta"]["version"] == "response-v2"

    def test_not_implemented_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """NOT_IMPLEMENTED responses should have correct structure."""
        from foundry_mcp.tools.git_integration import register_git_integration_tools

        register_git_integration_tools(mock_mcp, mock_config)

        task_create_commit = mock_mcp._tools["task-create-commit"]
        result = task_create_commit(spec_id="test-spec", task_id="task-1-1")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"
        assert result["success"] is False
        assert result["data"]["error_code"] == "NOT_IMPLEMENTED"
