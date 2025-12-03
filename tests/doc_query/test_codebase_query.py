"""Unit tests for the native DocsQuery helper."""

import json
from pathlib import Path

import pytest

from foundry_mcp.core.docs import DocsQuery


@pytest.fixture
def docs_path(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    payload = {
        "classes": [
            {
                "name": "Alpha",
                "file": "src/alpha.py",
                "line": 1,
                "bases": ["Base"],
                "docstring": "Alpha class",
                "methods": ["run"],
            },
        ],
        "functions": [
            {
                "name": "run_alpha",
                "file": "src/alpha.py",
                "line": 10,
                "docstring": "Top-level runner",
                "calls": ["helper"],
                "parameters": ["payload"],
            },
            {
                "name": "helper",
                "file": "src/utils.py",
                "line": 30,
                "docstring": "Utility helper",
                "calls": [],
                "parameters": [],
            },
        ],
        "dependencies": {
            "src.alpha": ["src.utils"],
        },
    }
    path = docs_dir / "codebase.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def docs_query(docs_path: Path) -> DocsQuery:
    query = DocsQuery(docs_path)
    assert query.load()
    return query


def test_find_class_exact_match(docs_query: DocsQuery):
    response = docs_query.find_class("Alpha")
    assert response.success
    assert response.results[0].name == "Alpha"
    assert response.results[0].file_path == "src/alpha.py"


def test_find_function_fuzzy_match(docs_query: DocsQuery):
    response = docs_query.find_function("help", exact=False)
    assert response.success
    assert any(res.name == "helper" for res in response.results)


def test_trace_calls_returns_callers_and_callees(docs_query: DocsQuery):
    response = docs_query.trace_calls("helper", direction="callers")
    assert response.success
    callers = {(entry.caller, entry.callee) for entry in response.results}
    assert ("run_alpha", "helper") in callers


def test_get_scope_plan_mode_lists_members(docs_query: DocsQuery):
    response = docs_query.get_scope("src/alpha.py", mode="plan")
    assert response.success
    scope = response.results[0]
    assert scope["classes"][0]["name"] == "Alpha"
    assert scope["functions"][0]["name"] == "run_alpha"


def test_get_refactor_candidates_includes_high_callers(docs_query: DocsQuery):
    candidates = docs_query.get_refactor_candidates(min_callers=1)
    assert candidates.success
    names = [candidate["name"] for candidate in candidates.results]
    assert "helper" in names or "run_alpha" in names
