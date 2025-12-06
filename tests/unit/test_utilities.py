"""
Unit tests for foundry_mcp.tools.utilities module.

Tests the utility tools for SDD cache management and schema export.
These tools use direct Python API calls instead of CLI subprocess calls.
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


# =============================================================================
# sdd-cache-manage Tool Tests
# =============================================================================


class TestSddCacheManage:
    """Test the sdd-cache-manage tool."""

    def test_invalid_action_returns_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return validation error for invalid action."""
        from foundry_mcp.tools.utilities import register_utility_tools

        register_utility_tools(mock_mcp, mock_config)

        cache_manage = mock_mcp._tools["sdd-cache-manage"]
        result = cache_manage(action="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"
        assert result["data"].get("error_type") == "validation"

    def test_invalid_review_type_returns_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return validation error for invalid review_type."""
        from foundry_mcp.tools.utilities import register_utility_tools

        register_utility_tools(mock_mcp, mock_config)

        cache_manage = mock_mcp._tools["sdd-cache-manage"]
        result = cache_manage(action="clear", review_type="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_cache_info_action(self, mock_mcp, mock_config, assert_response_contract):
        """Should return cache stats for info action."""
        from foundry_mcp.tools.utilities import register_utility_tools
        from foundry_mcp.core.cache import CacheManager

        # Mock the CacheManager
        mock_stats = {
            "total_entries": 10,
            "active_entries": 5,
            "expired_entries": 5,
            "total_size_bytes": 1024,
        }

        with patch.object(CacheManager, "get_stats", return_value=mock_stats):
            register_utility_tools(mock_mcp, mock_config)

            cache_manage = mock_mcp._tools["sdd-cache-manage"]
            result = cache_manage(action="info")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["action"] == "info"
        assert "cache" in result["data"]

    def test_cache_clear_action(self, mock_mcp, mock_config, assert_response_contract):
        """Should clear cache entries for clear action."""
        from foundry_mcp.tools.utilities import register_utility_tools
        from foundry_mcp.core.cache import CacheManager

        with patch.object(CacheManager, "clear", return_value=5):
            register_utility_tools(mock_mcp, mock_config)

            cache_manage = mock_mcp._tools["sdd-cache-manage"]
            result = cache_manage(action="clear")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["action"] == "clear"
        assert "entries_deleted" in result["data"]

    def test_cache_clear_with_filters(self, mock_mcp, mock_config, assert_response_contract):
        """Should pass filters to cache clear."""
        from foundry_mcp.tools.utilities import register_utility_tools
        from foundry_mcp.core.cache import CacheManager

        with patch.object(CacheManager, "clear", return_value=3) as mock_clear:
            register_utility_tools(mock_mcp, mock_config)

            cache_manage = mock_mcp._tools["sdd-cache-manage"]
            result = cache_manage(
                action="clear",
                spec_id="test-spec",
                review_type="fidelity"
            )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["filters"]["spec_id"] == "test-spec"
        assert result["data"]["filters"]["review_type"] == "fidelity"


# =============================================================================
# spec-schema-export Tool Tests
# =============================================================================


class TestSpecSchemaExport:
    """Test the spec-schema-export tool."""

    def test_invalid_schema_type_returns_error(self, mock_mcp, mock_config, assert_response_contract):
        """Should return validation error for invalid schema_type."""
        from foundry_mcp.tools.utilities import register_utility_tools

        register_utility_tools(mock_mcp, mock_config)

        schema_export = mock_mcp._tools["spec-schema-export"]
        result = schema_export(schema_type="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Test that all tools are properly registered."""

    def test_all_tools_registered(self, mock_mcp, mock_config):
        """All utility tools should be registered with the MCP server."""
        from foundry_mcp.tools.utilities import register_utility_tools

        register_utility_tools(mock_mcp, mock_config)

        expected_tools = [
            "sdd-cache-manage",
            "spec-schema-export",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.utilities import register_utility_tools

        register_utility_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_validation_error_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """Validation error responses should have correct structure."""
        from foundry_mcp.tools.utilities import register_utility_tools

        register_utility_tools(mock_mcp, mock_config)

        cache_manage = mock_mcp._tools["sdd-cache-manage"]
        result = cache_manage(action="invalid")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_success_response_has_required_fields(self, mock_mcp, mock_config, assert_response_contract):
        """Success responses should have all required fields."""
        from foundry_mcp.tools.utilities import register_utility_tools
        from foundry_mcp.core.cache import CacheManager

        mock_stats = {
            "total_entries": 10,
            "active_entries": 5,
        }

        with patch.object(CacheManager, "get_stats", return_value=mock_stats):
            register_utility_tools(mock_mcp, mock_config)

            cache_manage = mock_mcp._tools["sdd-cache-manage"]
            result = cache_manage(action="info")

        # Validate structure
        assert result["success"] is True
        assert result["error"] is None
        assert result["meta"]["version"] == "response-v2"
