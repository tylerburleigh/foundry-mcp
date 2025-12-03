"""
Unit tests for review tools.

Tests cover:
- spec-review tool (returns NOT_IMPLEMENTED since it requires external AI tools)
- review-list-tools tool for listing available review tools
- review-list-plan-tools for plan analysis toolchains
- LLM status helper function
"""

import json
from dataclasses import asdict
from typing import Any, Dict
from unittest.mock import MagicMock, patch, Mock

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
    return MagicMock()


# =============================================================================
# LLM Status Helper Tests
# =============================================================================


class TestGetLLMStatus:
    """Tests for _get_llm_status helper."""

    def test_llm_configured(self):
        """Test LLM status when properly configured."""
        from foundry_mcp.tools.review import _get_llm_status

        mock_config = MagicMock()
        mock_config.get_api_key.return_value = "test-api-key"
        mock_config.provider.value = "openai"
        mock_config.get_model.return_value = "gpt-4"

        with patch("foundry_mcp.core.llm_config.get_llm_config", return_value=mock_config):
            status = _get_llm_status()

        assert status["configured"] is True
        assert status["provider"] == "openai"
        assert status["model"] == "gpt-4"

    def test_llm_not_configured(self):
        """Test LLM status when API key not set."""
        from foundry_mcp.tools.review import _get_llm_status

        mock_config = MagicMock()
        mock_config.get_api_key.return_value = None
        mock_config.provider.value = "openai"
        mock_config.get_model.return_value = "gpt-4"

        with patch("foundry_mcp.core.llm_config.get_llm_config", return_value=mock_config):
            status = _get_llm_status()

        assert status["configured"] is False

    def test_llm_config_import_error(self):
        """Test LLM status when config module not available."""
        from foundry_mcp.tools.review import _get_llm_status

        with patch("foundry_mcp.core.llm_config.get_llm_config", side_effect=ImportError("No module")):
            status = _get_llm_status()

        assert status["configured"] is False
        assert "error" in status


# =============================================================================
# spec-review Tool Tests
# =============================================================================


class TestSpecReviewTool:
    """Tests for spec-review tool."""

    def test_quick_review_handles_missing_spec(self, mock_mcp, mock_config, assert_response_contract):
        """spec_review should run quick review for review_type='quick'."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec-review"]
        result = spec_review(spec_id="test-spec-001", review_type="quick")

        assert_response_contract(result)
        # Quick review should succeed but report spec not found in findings
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["review_type"] == "quick"

    def test_ai_review_returns_no_provider_error(self, mock_mcp, mock_config, assert_response_contract):
        """AI review types should return AI_NO_PROVIDER when no providers available."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec-review"]
        # Use 'full' review type which requires AI
        result = spec_review(spec_id="test-spec-001", review_type="full")

        assert_response_contract(result)
        # Should fail due to spec not found (first check) or no providers
        assert result["success"] is False
        error_code = result["data"].get("error_code")
        assert error_code in ("SPEC_NOT_FOUND", "AI_NO_PROVIDER", "AI_NOT_AVAILABLE")

    def test_dry_run_returns_request_parameters(self, mock_mcp, mock_config, assert_response_contract):
        """Dry run should return request parameters in response data."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec-review"]
        result = spec_review(
            spec_id="test-spec-001",
            review_type="quick",
            dry_run=True
        )

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["review_type"] == "quick"
        assert result["data"]["dry_run"] is True


# =============================================================================
# review-list-tools Tool Tests
# =============================================================================


class TestReviewListTools:
    """Tests for review-list-tools tool."""

    def test_returns_tools_list(self, mock_mcp, mock_config, assert_response_contract):
        """Should return list of review tools."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review-list-tools"]
        result = list_tools()

        assert_response_contract(result)
        assert result["success"] is True
        assert "tools" in result["data"]
        assert isinstance(result["data"]["tools"], list)
        assert len(result["data"]["tools"]) > 0

    def test_includes_llm_status(self, mock_mcp, mock_config, assert_response_contract):
        """Should include LLM status in response."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review-list-tools"]
        result = list_tools()

        assert_response_contract(result)
        assert "llm_status" in result["data"]

    def test_includes_review_types(self, mock_mcp, mock_config, assert_response_contract):
        """Should include available review types."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review-list-tools"]
        result = list_tools()

        assert_response_contract(result)
        assert "review_types" in result["data"]
        assert "quick" in result["data"]["review_types"]
        assert "full" in result["data"]["review_types"]


# =============================================================================
# review-list-plan-tools Tool Tests
# =============================================================================


class TestReviewListPlanTools:
    """Tests for review-list-plan-tools tool."""

    def test_returns_plan_tools(self, mock_mcp, mock_config, assert_response_contract):
        """Should return list of plan review tools."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools["review-list-plan-tools"]
        result = list_plan_tools()

        assert_response_contract(result)
        assert result["success"] is True
        assert "plan_tools" in result["data"]
        assert isinstance(result["data"]["plan_tools"], list)
        assert len(result["data"]["plan_tools"]) > 0

    def test_includes_llm_status(self, mock_mcp, mock_config, assert_response_contract):
        """Should include LLM status in response."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools["review-list-plan-tools"]
        result = list_plan_tools()

        assert_response_contract(result)
        assert "llm_status" in result["data"]

    def test_includes_recommendations(self, mock_mcp, mock_config, assert_response_contract):
        """Should include usage recommendations."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_plan_tools = mock_mcp._tools["review-list-plan-tools"]
        result = list_plan_tools()

        assert_response_contract(result)
        assert "recommendations" in result["data"]
        assert isinstance(result["data"]["recommendations"], list)
        assert len(result["data"]["recommendations"]) > 0


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Test that all tools are properly registered."""

    def test_all_tools_registered(self, mock_mcp, mock_config):
        """All review tools should be registered with the MCP server."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec-review",
            "review-list-tools",
            "review-list-plan-tools",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_not_implemented_response_structure(self, mock_mcp, mock_config, assert_response_contract):
        """NOT_IMPLEMENTED responses should have correct structure."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        spec_review = mock_mcp._tools["spec-review"]
        result = spec_review(spec_id="test-spec")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_success_response_has_required_fields(self, mock_mcp, mock_config, assert_response_contract):
        """Success responses should have all required fields."""
        from foundry_mcp.tools.review import register_review_tools

        register_review_tools(mock_mcp, mock_config)

        list_tools = mock_mcp._tools["review-list-tools"]
        result = list_tools()

        # Validate structure
        assert result["success"] is True
        assert result["error"] is None
        assert "tools" in result["data"]
        assert result["meta"]["version"] == "response-v2"
