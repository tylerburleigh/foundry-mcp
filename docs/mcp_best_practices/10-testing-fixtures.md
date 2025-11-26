# 10. Comprehensive Testing & Fixtures

> Test contracts, helpers, and integrations with meaningful fixtures.

## Overview

Testing MCP tools requires validating response contracts, helper functions, and end-to-end integrations. Fixtures must stay current with schema evolution.

## Requirements

### MUST

- **Test response envelope structure** - validate all responses match schema
- **Test helper functions** - unit tests for serialization helpers
- **Update fixtures when schema changes** - prevent stale test data
- **Test error conditions** - not just happy paths

### SHOULD

- **Use property-based testing** - for input validation
- **Test contract compliance** - responses match spec
- **Test edge cases** - empty results, large payloads, timeouts
- **Maintain fixture freshness** - automated regeneration

### MAY

- **Use snapshot testing** - for complex response structures
- **Test performance** - latency, throughput
- **Use contract testing** - consumer-driven contracts

## Test Categories

### 1. Unit Tests (Helpers)

```python
# tests/test_responses.py
import pytest
from dataclasses import asdict
from foundry_mcp.core.responses import success_response, error_response

class TestSuccessResponse:
    """Test success_response helper."""

    def test_minimal_success(self):
        """Success response with no data."""
        response = asdict(success_response())

        assert response["success"] is True
        assert response["error"] is None
        assert response["data"] == {}
        assert response["meta"]["version"] == "response-v2"

    def test_success_with_data(self):
        """Success response with data payload."""
        response = asdict(success_response(data={"key": "value"}))

        assert response["data"] == {"key": "value"}

    def test_success_with_warnings(self):
        """Success response with warnings."""
        response = asdict(success_response(
            data={},
            warnings=["Warning 1", "Warning 2"]
        ))

        assert response["meta"]["warnings"] == ["Warning 1", "Warning 2"]

    def test_success_with_pagination(self):
        """Success response with pagination metadata."""
        response = asdict(success_response(
            data={"items": []},
            pagination={
                "cursor": "abc123",
                "has_more": True
            }
        ))

        assert response["meta"]["pagination"]["cursor"] == "abc123"
        assert response["meta"]["pagination"]["has_more"] is True


class TestErrorResponse:
    """Test error_response helper."""

    def test_minimal_error(self):
        """Error response with just message."""
        response = asdict(error_response(error="Something failed"))

        assert response["success"] is False
        assert response["error"] == "Something failed"
        assert response["data"] == {}
        assert response["meta"]["version"] == "response-v2"

    def test_error_with_details(self):
        """Error response with additional data."""
        response = asdict(error_response(
            error="Validation failed",
            data={"validation_errors": [{"field": "email"}]}
        ))

        assert response["data"]["validation_errors"][0]["field"] == "email"
```

### 2. Contract Tests

```python
# tests/test_contracts.py
import pytest
import jsonschema
from pathlib import Path
import json

# Load response schema
RESPONSE_SCHEMA = json.loads(
    Path("schemas/response-v2.schema.json").read_text()
)

def validate_response(response: dict):
    """Validate response against schema."""
    jsonschema.validate(response, RESPONSE_SCHEMA)

class TestToolContracts:
    """Test that tools return valid responses."""

    def test_get_user_success_contract(self):
        """get_user success response matches contract."""
        response = get_user(user_id="usr_123")

        validate_response(response)
        assert response["success"] is True
        assert "user" in response["data"]

    def test_get_user_not_found_contract(self):
        """get_user error response matches contract."""
        response = get_user(user_id="usr_nonexistent")

        validate_response(response)
        assert response["success"] is False
        assert "error" in response
        assert response["error"] is not None

    def test_list_users_pagination_contract(self):
        """list_users pagination matches contract."""
        response = list_users(limit=10)

        validate_response(response)
        if response["meta"].get("pagination"):
            pagination = response["meta"]["pagination"]
            assert "has_more" in pagination
            assert isinstance(pagination["has_more"], bool)
```

### 3. Integration Tests

```python
# tests/integration/test_mcp_tools.py
import pytest
from foundry_mcp.server import MCPServer

@pytest.fixture
def server():
    """Create test MCP server."""
    return MCPServer(config={"test_mode": True})

class TestMCPIntegration:
    """Integration tests for MCP tools."""

    def test_tool_discovery(self, server):
        """Server exposes tools correctly."""
        tools = server.list_tools()

        assert len(tools) > 0
        for tool in tools:
            assert "name" in tool
            assert "description" in tool

    def test_tool_invocation(self, server):
        """Tools can be invoked and return valid responses."""
        result = server.invoke_tool("get_user", {"user_id": "usr_123"})

        assert "success" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_error_handling(self, server):
        """Errors are handled gracefully."""
        result = server.invoke_tool("get_user", {"user_id": ""})

        assert result["success"] is False
        assert result["error"] is not None
```

### 4. Property-Based Tests

```python
# tests/test_properties.py
from hypothesis import given, strategies as st
import pytest

class TestInputValidation:
    """Property-based tests for input validation."""

    @given(st.text(max_size=1000))
    def test_string_input_never_crashes(self, text):
        """Tool handles arbitrary string input without crashing."""
        try:
            result = process_text(text=text)
            # Should return valid response structure
            assert "success" in result
            assert "meta" in result
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

    @given(st.integers())
    def test_integer_bounds(self, value):
        """Tool handles integer edge cases."""
        result = set_count(count=value)

        if value < 0 or value > 1000:
            assert result["success"] is False
        else:
            assert result["success"] is True

    @given(st.lists(st.text(), max_size=100))
    def test_list_input(self, items):
        """Tool handles list inputs of various sizes."""
        result = process_batch(items=items)

        assert "success" in result
        if result["success"]:
            assert "processed" in result["data"]
```

## Fixture Management

### Fixture Structure

```
tests/
├── fixtures/
│   ├── responses/
│   │   ├── get_user_success.json
│   │   ├── get_user_not_found.json
│   │   └── list_users_paginated.json
│   ├── inputs/
│   │   ├── valid_user_input.json
│   │   └── invalid_inputs.json
│   └── mocks/
│       └── database_responses.json
```

### Fixture Generation Script

```python
#!/usr/bin/env python3
"""Regenerate test fixtures from current schema."""

import json
from pathlib import Path
from dataclasses import asdict
from foundry_mcp.core.responses import success_response, error_response

FIXTURES_DIR = Path("tests/fixtures/responses")

def generate_fixtures():
    """Generate canonical fixture files."""
    fixtures = {
        "get_user_success.json": asdict(success_response(
            data={
                "user": {
                    "id": "usr_fixture123",
                    "name": "Test User",
                    "email": "test@example.com"
                }
            }
        )),

        "get_user_not_found.json": asdict(error_response(
            error="User 'usr_nonexistent' not found",
            data={"error_code": "USER_NOT_FOUND"}
        )),

        "list_users_paginated.json": asdict(success_response(
            data={
                "users": [
                    {"id": "usr_1", "name": "User 1"},
                    {"id": "usr_2", "name": "User 2"}
                ]
            },
            pagination={
                "cursor": "eyJvZmZzZXQiOjJ9",
                "has_more": True,
                "total_count": 100
            }
        )),
    }

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for filename, data in fixtures.items():
        filepath = FIXTURES_DIR / filename
        filepath.write_text(json.dumps(data, indent=2))
        print(f"Generated: {filepath}")

if __name__ == "__main__":
    generate_fixtures()
```

### Fixture Freshness Check

```python
# tests/conftest.py
import pytest
import json
from pathlib import Path
from foundry_mcp.core.responses import success_response
from dataclasses import asdict

@pytest.fixture(autouse=True)
def check_fixture_schema_version():
    """Verify fixtures use current schema version."""
    fixtures_dir = Path("tests/fixtures/responses")
    current_version = asdict(success_response())["meta"]["version"]

    for fixture_file in fixtures_dir.glob("*.json"):
        data = json.loads(fixture_file.read_text())
        if "meta" in data and "version" in data["meta"]:
            fixture_version = data["meta"]["version"]
            if fixture_version != current_version:
                pytest.fail(
                    f"Fixture {fixture_file} uses version '{fixture_version}' "
                    f"but current version is '{current_version}'. "
                    f"Run 'python scripts/generate_fixtures.py' to update."
                )
```

## Snapshot Testing

```python
# tests/test_snapshots.py
import pytest
import json

class TestSnapshots:
    """Snapshot tests for complex responses."""

    def test_complex_response_snapshot(self, snapshot):
        """Complex response matches snapshot."""
        response = generate_complex_report(report_id="rpt_123")

        # snapshot library compares and updates
        snapshot.assert_match(
            json.dumps(response, indent=2, sort_keys=True),
            "complex_report.json"
        )

    def test_error_response_snapshot(self, snapshot):
        """Error response format matches snapshot."""
        response = get_user(user_id="invalid!")

        snapshot.assert_match(
            json.dumps(response, indent=2, sort_keys=True),
            "validation_error.json"
        )
```

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[test]"

      - name: Check fixture freshness
        run: python scripts/check_fixtures.py

      - name: Run unit tests
        run: pytest tests/unit -v

      - name: Run contract tests
        run: pytest tests/contracts -v

      - name: Run integration tests
        run: pytest tests/integration -v

      - name: Run property tests
        run: pytest tests/properties -v --hypothesis-seed=0

      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

## Anti-Patterns

### Don't: Test Implementation, Not Contract

```python
# Bad: Testing internal implementation
def test_bad():
    result = get_user("usr_123")
    assert result._internal_cache_hit is True  # Implementation detail!

# Good: Testing contract
def test_good():
    result = get_user("usr_123")
    assert result["success"] is True
    assert "user" in result["data"]
```

### Don't: Hardcode Schema in Tests

```python
# Bad: Hardcoded version
def test_bad():
    assert response["meta"]["version"] == "response-v2"  # Will break on upgrade

# Good: Use constant or schema
def test_good():
    from foundry_mcp.core.responses import CURRENT_VERSION
    assert response["meta"]["version"] == CURRENT_VERSION
```

### Don't: Ignore Edge Cases

```python
# Bad: Only happy path
def test_bad():
    result = list_users()
    assert result["success"] is True

# Good: Cover edge cases
def test_empty_list():
    result = list_users(filter="nonexistent")
    assert result["success"] is True
    assert result["data"]["users"] == []

def test_max_limit():
    result = list_users(limit=10000)
    assert result["success"] is False  # Exceeds max
```

## Related Documents

- [Serialization Helpers](./03-serialization-helpers.md) - Helpers to test
- [Spec-Driven Development](./09-spec-driven-development.md) - Spec-based tests
- [Error Semantics](./07-error-semantics.md) - Error case testing

---

**Navigation:** [← Spec-Driven Development](./09-spec-driven-development.md) | [Index](./README.md) | [Next: AI/LLM Integration →](./11-ai-llm-integration.md)
