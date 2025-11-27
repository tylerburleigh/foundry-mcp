"""
Property-based tests for input validation using Hypothesis.

Tests that arbitrary inputs don't crash validation and errors are handled properly.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import json

from foundry_mcp.core.validation import (
    validate_spec_input,
    validate_spec,
    ValidationResult,
    Diagnostic,
    VALID_STATUSES,
    VALID_NODE_TYPES,
    VALID_VERIFICATION_TYPES,
    VALID_TASK_CATEGORIES,
)
from foundry_mcp.core.security import (
    MAX_INPUT_SIZE,
    MAX_ARRAY_LENGTH,
    MAX_STRING_LENGTH,
    MAX_NESTED_DEPTH,
)


# Custom strategies for valid data generation

@st.composite
def valid_node_id(draw):
    """Generate valid node IDs."""
    prefix = draw(st.sampled_from(["task", "phase", "group", "subtask", "verify"]))
    suffix = draw(st.text(
        alphabet="0123456789-",
        min_size=1,
        max_size=20
    ))
    return f"{prefix}-{suffix}"


@st.composite
def valid_status(draw):
    """Generate valid status values."""
    return draw(st.sampled_from(list(VALID_STATUSES)))


@st.composite
def valid_node_type(draw):
    """Generate valid node types."""
    return draw(st.sampled_from(list(VALID_NODE_TYPES)))


@st.composite
def minimal_valid_node(draw):
    """Generate a minimal valid node structure."""
    return {
        "type": draw(valid_node_type()),
        "title": draw(st.text(min_size=1, max_size=100)),
        "status": draw(valid_status()),
        "parent": None,  # Will be set when building hierarchy
        "children": [],
        "total_tasks": 1,
        "completed_tasks": 0,
        "metadata": {},
    }


@st.composite
def valid_spec_id(draw):
    """Generate valid spec IDs."""
    feature = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=3,
        max_size=30
    ))
    year = draw(st.integers(min_value=2020, max_value=2030))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))
    seq = draw(st.integers(min_value=1, max_value=999))
    return f"{feature}-{year}-{month:02d}-{day:02d}-{seq:03d}"


@st.composite
def minimal_valid_spec(draw):
    """Generate a minimal valid spec structure."""
    spec_id = draw(valid_spec_id())
    return {
        "spec_id": spec_id,
        "generated": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:00:00Z",
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": draw(st.text(min_size=1, max_size=100)),
                "status": "pending",
                "parent": None,
                "children": [],
                "total_tasks": 0,
                "completed_tasks": 0,
                "metadata": {},
            }
        }
    }


class TestInputValidationRobustness:
    """Property tests ensuring validation never crashes on arbitrary input."""

    @given(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), max_size=1000))
    @settings(max_examples=100)
    def test_utf8_text_input_never_crashes(self, text):
        """Validation handles arbitrary UTF-8 text without crashing."""
        # Encode to bytes (UTF-8)
        data = text.encode('utf-8')
        result, error = validate_spec_input(data)

        # Must return a result - either parsed data or error
        assert result is not None or error is not None

        # If error, must be a valid ValidationResult
        if error is not None:
            assert isinstance(error, ValidationResult)
            assert not error.is_valid
            assert len(error.diagnostics) > 0

    @given(st.text(max_size=5000))
    @settings(max_examples=100)
    def test_text_input_never_crashes(self, text):
        """Validation handles arbitrary text without crashing."""
        result, error = validate_spec_input(text)

        # Must return a result
        assert result is not None or error is not None

        # If we got parsed data and it's a dict, validate_spec should not crash
        if result is not None and isinstance(result, dict):
            validation_result = validate_spec(result)
            assert isinstance(validation_result, ValidationResult)

    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.booleans(),
            st.none(),
            st.lists(st.text(max_size=50), max_size=10),
        ),
        max_size=20
    ))
    @settings(max_examples=50)
    def test_arbitrary_dict_never_crashes(self, data):
        """Validation handles arbitrary dictionary structures."""
        json_str = json.dumps(data)
        result, error = validate_spec_input(json_str)

        if result is not None:
            validation_result = validate_spec(result)
            assert isinstance(validation_result, ValidationResult)
            # Arbitrary dicts should fail validation (missing required fields)
            # but the function should not crash
            assert validation_result.diagnostics is not None


class TestSizeLimitEnforcement:
    """Property tests for size limit enforcement."""

    def test_oversized_input_rejected(self):
        """Inputs exceeding size limit are rejected with proper error."""
        # Create oversized UTF-8 input
        oversized_data = ("x" * (MAX_INPUT_SIZE + 100)).encode('utf-8')
        result, error = validate_spec_input(oversized_data)

        assert result is None
        assert error is not None
        assert not error.is_valid
        assert any(d.code == "INPUT_TOO_LARGE" for d in error.diagnostics)

    @given(st.integers(min_value=1, max_value=min(MAX_INPUT_SIZE, 100000)))
    @settings(max_examples=20)
    def test_size_limit_boundary(self, size):
        """Inputs at or below size limit are accepted for parsing."""
        # Create JSON of approximately the target size
        padding = "x" * max(0, size - 50)
        data = json.dumps({"padding": padding})

        if len(data.encode('utf-8')) <= MAX_INPUT_SIZE:
            result, error = validate_spec_input(data)
            # Should parse (may fail validation, but shouldn't fail size check)
            assert result is not None or (
                error is not None and
                not any(d.code == "INPUT_TOO_LARGE" for d in error.diagnostics)
            )


class TestMalformedJsonHandling:
    """Property tests for malformed JSON handling."""

    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_invalid_json_returns_error(self, text):
        """Invalid JSON returns proper error, never crashes."""
        # Skip if text happens to be valid JSON
        try:
            json.loads(text)
            assume(False)  # Skip valid JSON
        except json.JSONDecodeError:
            pass

        result, error = validate_spec_input(text)

        assert result is None
        assert error is not None
        assert not error.is_valid
        assert any(d.code == "INVALID_JSON" for d in error.diagnostics)

    @given(st.sampled_from([
        "{",
        "[",
        '{"key":}',
        '{"key": "value"',
        "{'key': 'value'}",  # Single quotes
        '{"key": undefined}',
        # Note: NaN is accepted by Python's json.loads (non-standard)
    ]))
    def test_common_json_errors(self, malformed):
        """Common JSON syntax errors are handled gracefully."""
        result, error = validate_spec_input(malformed)

        assert result is None
        assert error is not None
        assert not error.is_valid


class TestValidSpecProperties:
    """Property tests for valid spec structures."""

    @given(minimal_valid_spec())
    @settings(max_examples=50)
    def test_minimal_spec_validates(self, spec):
        """Minimal valid specs pass validation."""
        json_str = json.dumps(spec)
        result, error = validate_spec_input(json_str)

        assert result is not None
        assert error is None

        validation_result = validate_spec(result)
        # Should have no errors (may have warnings)
        assert validation_result.error_count == 0

    @given(minimal_valid_spec(), st.lists(valid_node_id(), min_size=1, max_size=10))
    @settings(max_examples=30)
    def test_spec_with_children_validates(self, spec, child_ids):
        """Specs with properly linked children validate."""
        # Add child nodes
        for child_id in child_ids:
            spec["hierarchy"][child_id] = {
                "type": "task",
                "title": f"Task {child_id}",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
            }
            spec["hierarchy"]["spec-root"]["children"].append(child_id)

        # Update counts
        spec["hierarchy"]["spec-root"]["total_tasks"] = len(child_ids)

        json_str = json.dumps(spec)
        result, error = validate_spec_input(json_str)

        assert result is not None
        validation_result = validate_spec(result)
        # Should validate without hierarchy errors
        hierarchy_errors = [
            d for d in validation_result.diagnostics
            if d.category == "hierarchy" and d.severity == "error"
        ]
        assert len(hierarchy_errors) == 0


class TestStatusValidation:
    """Property tests for status field validation."""

    @given(valid_status())
    @settings(max_examples=20)
    def test_valid_statuses_accepted(self, status):
        """Valid status values are accepted."""
        spec = {
            "spec_id": "test-2025-01-01-001",
            "generated": "2025-01-01T00:00:00Z",
            "last_updated": "2025-01-01T00:00:00Z",
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Test",
                    "status": status,
                    "parent": None,
                    "children": [],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {},
                }
            }
        }

        result = validate_spec(spec)
        status_errors = [
            d for d in result.diagnostics
            if d.code == "INVALID_STATUS"
        ]
        assert len(status_errors) == 0

    @given(st.text(min_size=1, max_size=50).filter(lambda x: x not in VALID_STATUSES))
    @settings(max_examples=50)
    def test_invalid_statuses_rejected(self, status):
        """Invalid status values produce diagnostics."""
        spec = {
            "spec_id": "test-2025-01-01-001",
            "generated": "2025-01-01T00:00:00Z",
            "last_updated": "2025-01-01T00:00:00Z",
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Test",
                    "status": status,
                    "parent": None,
                    "children": [],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {},
                }
            }
        }

        result = validate_spec(spec)
        status_errors = [
            d for d in result.diagnostics
            if d.code == "INVALID_STATUS"
        ]
        assert len(status_errors) > 0


class TestNodeTypeValidation:
    """Property tests for node type validation."""

    @given(valid_node_type())
    @settings(max_examples=20)
    def test_valid_types_accepted(self, node_type):
        """Valid node types are accepted."""
        spec = {
            "spec_id": "test-2025-01-01-001",
            "generated": "2025-01-01T00:00:00Z",
            "last_updated": "2025-01-01T00:00:00Z",
            "hierarchy": {
                "spec-root": {
                    "type": node_type,
                    "title": "Test",
                    "status": "pending",
                    "parent": None,
                    "children": [],
                    "total_tasks": 1 if node_type in {"task", "subtask", "verify"} else 0,
                    "completed_tasks": 0,
                    "metadata": {},
                }
            }
        }

        result = validate_spec(spec)
        type_errors = [
            d for d in result.diagnostics
            if d.code == "INVALID_NODE_TYPE"
        ]
        assert len(type_errors) == 0

    @given(st.text(min_size=1, max_size=50).filter(lambda x: x not in VALID_NODE_TYPES))
    @settings(max_examples=50)
    def test_invalid_types_rejected(self, node_type):
        """Invalid node types produce diagnostics."""
        spec = {
            "spec_id": "test-2025-01-01-001",
            "generated": "2025-01-01T00:00:00Z",
            "last_updated": "2025-01-01T00:00:00Z",
            "hierarchy": {
                "spec-root": {
                    "type": node_type,
                    "title": "Test",
                    "status": "pending",
                    "parent": None,
                    "children": [],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {},
                }
            }
        }

        result = validate_spec(spec)
        type_errors = [
            d for d in result.diagnostics
            if d.code == "INVALID_NODE_TYPE"
        ]
        assert len(type_errors) > 0


class TestDiagnosticStructure:
    """Property tests ensuring diagnostic structure is always valid."""

    @given(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), max_size=500))
    @settings(max_examples=50)
    def test_diagnostics_always_well_formed(self, text):
        """All diagnostics have required fields."""
        # Use UTF-8 encoded text
        data = text.encode('utf-8')
        result, error = validate_spec_input(data)

        diagnostics = []
        if error is not None:
            diagnostics.extend(error.diagnostics)

        if result is not None and isinstance(result, dict):
            validation_result = validate_spec(result)
            diagnostics.extend(validation_result.diagnostics)

        for diag in diagnostics:
            assert isinstance(diag, Diagnostic)
            assert diag.code is not None and len(diag.code) > 0
            assert diag.message is not None and len(diag.message) > 0
            assert diag.severity in {"error", "warning", "info"}
            assert diag.category is not None and len(diag.category) > 0

    @given(minimal_valid_spec())
    @settings(max_examples=30)
    def test_valid_spec_no_error_diagnostics(self, spec):
        """Valid specs produce no error-severity diagnostics."""
        result = validate_spec(spec)

        error_diagnostics = [
            d for d in result.diagnostics
            if d.severity == "error"
        ]
        assert len(error_diagnostics) == 0
        assert result.is_valid
