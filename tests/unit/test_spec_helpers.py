"""
Unit tests for spec helper tools.

Tests the spec-find-related-files tool and associated functionality.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestSpecFindRelatedFiles:
    """Tests for spec-find-related-files tool."""

    def test_successful_find_related_files(self):
        """Test successful find-related-files command execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "related_files": [
                {"path": "src/module/helper.py", "relationship": "import"},
                {"path": "tests/test_module.py", "relationship": "test"},
            ],
            "spec_references": [
                {"spec_id": "feature-123", "task_id": "task-1-1"},
            ],
        })
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            # Create mock MCP and config
            mock_mcp = MagicMock()
            mock_config = MagicMock()

            # Capture the decorated function
            captured_func = None

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    nonlocal captured_func
                    captured_func = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            # Call the captured function
            result = captured_func("src/module/main.py")

            assert result["success"] is True
            assert result["data"]["file_path"] == "src/module/main.py"
            assert len(result["data"]["related_files"]) == 2
            assert result["data"]["total_count"] == 2
            assert len(result["data"]["spec_references"]) == 1

    def test_find_related_files_empty_result(self):
        """Test find-related-files with no related files found."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_func = None

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    nonlocal captured_func
                    captured_func = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_func("src/unknown/file.py")

            assert result["success"] is True
            assert result["data"]["related_files"] == []
            assert result["data"]["total_count"] == 0

    def test_find_related_files_with_spec_id(self):
        """Test find-related-files with spec_id filter."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"related_files": [], "spec_references": []})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_func = None

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    nonlocal captured_func
                    captured_func = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_func("src/main.py", spec_id="feature-123")

            # Verify the command included spec_id
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--spec-id" in cmd
            assert "feature-123" in cmd

    def test_find_related_files_with_metadata(self):
        """Test find-related-files with include_metadata=True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"related_files": []})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_func = None

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    nonlocal captured_func
                    captured_func = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_func("src/main.py", include_metadata=True)

            assert result["success"] is True
            assert "metadata" in result["data"]
            assert "command" in result["data"]["metadata"]
            assert "exit_code" in result["data"]["metadata"]

    def test_find_related_files_command_failure(self):
        """Test find-related-files when command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "File not found"

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_func = None

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    nonlocal captured_func
                    captured_func = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_func("nonexistent/file.py")

            assert result["success"] is False
            assert "COMMAND_FAILED" in result["data"]["error_code"]
            assert "File not found" in result["error"]

    def test_find_related_files_timeout(self):
        """Test find-related-files timeout handling."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sdd", 30)):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_func = None

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    nonlocal captured_func
                    captured_func = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_func("large/project/file.py")

            assert result["success"] is False
            assert "TIMEOUT" in result["data"]["error_code"]

    def test_find_related_files_cli_not_found(self):
        """Test find-related-files when SDD CLI is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError("sdd")):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_func = None

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    nonlocal captured_func
                    captured_func = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_func("src/main.py")

            assert result["success"] is False
            assert "CLI_NOT_FOUND" in result["data"]["error_code"]


class TestResponseEnvelopeCompliance:
    """Tests to verify response envelope compliance."""

    def test_success_response_has_required_fields(self):
        """Test success responses include all required fields."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_func = None

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    nonlocal captured_func
                    captured_func = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_func("src/main.py")

            # Verify response-v2 envelope
            assert "success" in result
            assert "data" in result
            assert "error" in result
            assert "meta" in result
            assert result["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(self):
        """Test error responses include all required fields."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"

        with patch("subprocess.run", return_value=mock_result):
            from foundry_mcp.tools.spec_helpers import register_spec_helper_tools

            mock_mcp = MagicMock()
            mock_config = MagicMock()
            captured_func = None

            def capture_decorator(mcp, canonical_name):
                def decorator(func):
                    nonlocal captured_func
                    captured_func = func
                    return func
                return decorator

            with patch("foundry_mcp.tools.spec_helpers.canonical_tool", capture_decorator):
                with patch("foundry_mcp.tools.spec_helpers.audit_log"):
                    register_spec_helper_tools(mock_mcp, mock_config)

            result = captured_func("src/main.py")

            # Verify response-v2 envelope
            assert result["success"] is False
            assert "data" in result
            assert "error_code" in result["data"]
            assert "error_type" in result["data"]
            assert "remediation" in result["data"]
            assert result["error"] is not None
            assert result["meta"]["version"] == "response-v2"
