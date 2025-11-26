"""
Tests for response helper functions and standard format validation.

Verifies that the response contract is properly implemented across all tools.
"""

import pytest
from foundry_mcp.core.responses import (
    ToolResponse,
    success_response,
    error_response,
)


class TestToolResponse:
    """Tests for the ToolResponse dataclass."""

    def test_success_response_structure(self):
        """Test that success responses have correct structure."""
        response = ToolResponse(
            success=True, data={"spec_id": "test-spec", "count": 5}, error=None
        )
        assert response.success is True
        assert response.data == {"spec_id": "test-spec", "count": 5}
        assert response.error is None

    def test_error_response_structure(self):
        """Test that error responses have correct structure."""
        response = ToolResponse(success=False, data={}, error="Something went wrong")
        assert response.success is False
        assert response.data == {}
        assert response.error == "Something went wrong"

    def test_default_data_is_empty_dict(self):
        """Test that data defaults to empty dict."""
        response = ToolResponse(success=True, error=None)
        assert response.data == {}

    def test_data_is_mutable(self):
        """Test that data can be modified."""
        response = ToolResponse(success=True, data={}, error=None)
        response.data["new_key"] = "value"
        assert response.data == {"new_key": "value"}


class TestSuccessResponse:
    """Tests for the success_response helper function."""

    def test_creates_success_true(self):
        """Test that success_response sets success=True."""
        response = success_response()
        assert response.success is True

    def test_sets_error_none(self):
        """Test that success_response sets error=None."""
        response = success_response(spec_id="test")
        assert response.error is None

    def test_passes_kwargs_to_data(self):
        """Test that kwargs are included in data."""
        response = success_response(
            spec_id="my-spec", count=10, tasks=["task-1", "task-2"]
        )
        assert response.data == {
            "spec_id": "my-spec",
            "count": 10,
            "tasks": ["task-1", "task-2"],
        }

    def test_data_argument_merges_with_kwargs(self):
        """Test that explicit data dict merges with additional kwargs."""
        response = success_response(
            data={"spec_id": "my-spec", "count": 5}, tasks=["task-1"], status="active"
        )
        assert response.data == {
            "spec_id": "my-spec",
            "count": 5,
            "tasks": ["task-1"],
            "status": "active",
        }

    def test_empty_response(self):
        """Test success_response with no arguments."""
        response = success_response()
        assert response.success is True
        assert response.data == {}
        assert response.error is None

    def test_nested_data_structures(self):
        """Test that nested data structures are preserved."""
        response = success_response(
            spec={"id": "test", "title": "Test Spec"},
            metadata={"version": "1.0", "author": "test"},
        )
        assert response.data["spec"]["id"] == "test"
        assert response.data["metadata"]["version"] == "1.0"


class TestErrorResponse:
    """Tests for the error_response helper function."""

    def test_creates_success_false(self):
        """Test that error_response sets success=False."""
        response = error_response("Test error")
        assert response.success is False

    def test_sets_error_message(self):
        """Test that error_response sets the error message."""
        response = error_response("Spec not found")
        assert response.error == "Spec not found"

    def test_data_is_empty_dict(self):
        """Test that error_response data is empty dict."""
        response = error_response("Test error")
        assert response.data == {}

    def test_preserves_error_message_details(self):
        """Test that detailed error messages are preserved."""
        message = "Task 'task-1-2' not found in spec 'my-spec'"
        response = error_response(message)
        assert response.error == message

    def test_structured_error_payload_and_meta(self):
        """Test that structured error data and meta fields are supported."""
        response = error_response(
            "Validation failed",
            error_code="INVALID_INPUT",
            error_type="validation",
            remediation="Provide a non-empty spec_id",
            details={"field": "spec_id"},
            data={"attempted_spec_id": ""},
            request_id="req_999",
            rate_limit={"limit": 100, "remaining": 0},
            telemetry={"duration_ms": 12},
            meta={"trace_id": "trace_error"},
        )
        assert response.data["error_code"] == "INVALID_INPUT"
        assert response.data["error_type"] == "validation"
        assert response.data["remediation"] == "Provide a non-empty spec_id"
        assert response.data["details"] == {"field": "spec_id"}
        assert response.data["attempted_spec_id"] == ""
        assert response.meta["request_id"] == "req_999"
        assert response.meta["rate_limit"]["remaining"] == 0
        assert response.meta["telemetry"]["duration_ms"] == 12
        assert response.meta["trace_id"] == "trace_error"


class TestResponseContractCompliance:
    """Tests verifying response contract compliance."""

    def test_success_response_has_required_fields(self):
        """Test that success responses include all required fields."""
        response = success_response(test_data="value")
        assert hasattr(response, "success")
        assert hasattr(response, "data")
        assert hasattr(response, "error")

    def test_error_response_has_required_fields(self):
        """Test that error responses include all required fields."""
        response = error_response("Test error")
        assert hasattr(response, "success")
        assert hasattr(response, "data")
        assert hasattr(response, "error")

    def test_success_response_types(self):
        """Test that success response fields have correct types."""
        response = success_response(count=5)
        assert isinstance(response.success, bool)
        assert isinstance(response.data, dict)
        assert response.error is None

    def test_error_response_types(self):
        """Test that error response fields have correct types."""
        response = error_response("Test error")
        assert isinstance(response.success, bool)
        assert isinstance(response.data, dict)
        assert isinstance(response.error, str)

    def test_meta_fields_convention(self):
        """Test that metadata helpers populate meta.* fields."""
        response = success_response(
            spec_id="test",
            warnings=["Deprecated field used"],
            pagination={"cursor": "abc123", "has_more": True},
            rate_limit={"limit": 100, "remaining": 42},
            telemetry={"duration_ms": 21},
            request_id="req_123",
            meta={"trace_id": "trace_456"},
        )
        assert response.meta["warnings"] == ["Deprecated field used"]
        assert response.meta["pagination"]["cursor"] == "abc123"
        assert response.meta["rate_limit"]["limit"] == 100
        assert response.meta["telemetry"]["duration_ms"] == 21
        assert response.meta["request_id"] == "req_123"
        assert response.meta["trace_id"] == "trace_456"

    def test_empty_result_is_success(self):
        """Test that empty results are still success=True."""
        # Following the contract: valid query, empty result
        response = success_response(tasks=[], count=0)
        assert response.success is True
        assert response.data["tasks"] == []
        assert response.data["count"] == 0

    def test_not_found_is_error(self):
        """Test that not found scenarios return error."""
        response = error_response("Spec not found: my-spec")
        assert response.success is False
        assert "not found" in response.error.lower()


class TestMetaVersionCompliance:
    """Tests verifying meta.version field presence per response-v2 contract."""

    def test_tool_response_has_meta_field(self):
        """Test that ToolResponse dataclass has meta field."""
        response = ToolResponse(success=True)
        assert hasattr(response, "meta")

    def test_tool_response_meta_default_version(self):
        """Test that meta field defaults to version='response-v2'."""
        response = ToolResponse(success=True)
        assert response.meta == {"version": "response-v2"}

    def test_success_response_includes_meta_version(self):
        """Test that success_response includes meta.version='response-v2'."""
        response = success_response(spec_id="test")
        assert response.meta is not None
        assert "version" in response.meta
        assert response.meta["version"] == "response-v2"

    def test_error_response_includes_meta_version(self):
        """Test that error_response includes meta.version='response-v2'."""
        response = error_response("Test error")
        assert response.meta is not None
        assert "version" in response.meta
        assert response.meta["version"] == "response-v2"

    def test_meta_preserved_when_data_set(self):
        """Test that meta.version is preserved when data is provided."""
        response = success_response(
            spec_id="my-spec", count=10, tasks=["task-1", "task-2"]
        )
        assert response.meta["version"] == "response-v2"
        assert response.data["spec_id"] == "my-spec"

    def test_meta_in_asdict_output(self):
        """Test that meta field appears in asdict() serialization."""
        from dataclasses import asdict

        response = success_response(test="value")
        response_dict = asdict(response)
        assert "meta" in response_dict
        assert response_dict["meta"]["version"] == "response-v2"

    def test_error_meta_in_asdict_output(self):
        """Test that error response meta field appears in asdict()."""
        from dataclasses import asdict

        response = error_response("Something failed")
        response_dict = asdict(response)
        assert "meta" in response_dict
        assert response_dict["meta"]["version"] == "response-v2"
        assert response_dict["success"] is False
        assert response_dict["error"] == "Something failed"

    def test_full_response_structure_compliance(self):
        """Test complete response-v2 structure: success, data, error, meta."""
        from dataclasses import asdict

        response = success_response(result="ok")
        response_dict = asdict(response)

        # All required fields present
        assert "success" in response_dict
        assert "data" in response_dict
        assert "error" in response_dict
        assert "meta" in response_dict

        # Correct values
        assert response_dict["success"] is True
        assert response_dict["data"] == {"result": "ok"}
        assert response_dict["error"] is None
        assert response_dict["meta"] == {"version": "response-v2"}
