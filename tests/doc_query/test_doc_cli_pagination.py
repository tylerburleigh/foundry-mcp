"""CLI pagination tests for doc commands."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from foundry_mcp.cli.main import cli


class StubDocsQuery:
    """Minimal DocsQuery stub for pagination tests."""

    def __init__(self, workspace=None):
        self.workspace = workspace
        self._classes_by_name = {
            "Alpha": {"file": "src/module_a.py", "line": 1},
            "Beta": {"file": "src/module_b.py", "line": 10},
        }
        self._functions_by_name = {
            "alpha": {"file": "src/module_a.py", "line": 5, "signature": "alpha()"},
            "beta": {"file": "src/module_a.py", "line": 25, "signature": "beta()"},
            "gamma": {"file": "src/module_b.py", "line": 15, "signature": "gamma()"},
        }
        self._functions_by_file = {
            "src/module_a.py": [
                {"name": "alpha", "line": 5, "signature": "alpha()"},
                {"name": "beta", "line": 25, "signature": "beta()"},
            ],
            "src/module_b.py": [
                {"name": "gamma", "line": 15, "signature": "gamma()"},
            ],
        }
        self._classes_by_file = {
            "src/module_a.py": [{"name": "Alpha"}],
            "src/module_b.py": [{"name": "Beta"}],
        }

    def load(self) -> bool:  # pragma: no cover - simple stub
        return True

    def find_functions_in_file(self, file_path: str) -> SimpleNamespace:
        rows = []
        for entry in self._functions_by_file.get(file_path, []):
            rows.append(
                SimpleNamespace(
                    name=entry["name"],
                    file_path=file_path,
                    line_number=entry["line"],
                    data={"signature": entry["signature"]},
                )
            )
        return SimpleNamespace(results=rows)

    def search(
        self, query: str, entity_types=None, max_results=None
    ) -> SimpleNamespace:
        results = [
            SimpleNamespace(
                name="alpha",
                entity_type="function",
                file_path="src/module_a.py",
                line_number=5,
                relevance_score=0.9,
                data={"signature": "alpha()"},
            ),
            SimpleNamespace(
                name="beta",
                entity_type="function",
                file_path="src/module_a.py",
                line_number=25,
                relevance_score=0.8,
                data={"signature": "beta()"},
            ),
        ]
        if max_results is not None:
            results = results[:max_results]
        return SimpleNamespace(success=True, results=results)

    def get_refactor_candidates(
        self, min_callers: int = 3, min_complexity: int = 0
    ) -> SimpleNamespace:
        candidates = [
            {
                "name": "alpha",
                "type": "function",
                "file_path": "src/module_a.py",
                "line": 5,
                "caller_count": 10,
                "callee_count": 2,
                "complexity": 4,
                "reasons": ["high_callers"],
            },
            {
                "name": "beta",
                "type": "function",
                "file_path": "src/module_a.py",
                "line": 25,
                "caller_count": 8,
                "callee_count": 1,
                "complexity": 3,
                "reasons": ["high_callers"],
            },
        ]
        return SimpleNamespace(success=True, results=candidates)


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def specs_dir(tmp_path):
    specs = tmp_path / "specs"
    specs.mkdir()
    return specs


@pytest.fixture(autouse=True)
def patch_docs(monkeypatch):
    monkeypatch.setattr("foundry_mcp.core.docs.DocsQuery", StubDocsQuery)


def _load_output(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_doc_list_modules_paginates(cli_runner, specs_dir):
    result = cli_runner.invoke(
        cli,
        ["--specs-dir", str(specs_dir), "doc", "list-modules", "--limit", "1"],
    )
    first = _load_output(result)
    cursor = first["meta"]["pagination"]["cursor"]
    assert cursor
    first_module = first["data"]["modules"][0]["file_path"]

    second = cli_runner.invoke(
        cli,
        [
            "--specs-dir",
            str(specs_dir),
            "doc",
            "list-modules",
            "--limit",
            "1",
            "--cursor",
            cursor,
        ],
    )
    follow = _load_output(second)
    assert follow["data"]["modules"][0]["file_path"] != first_module


def test_doc_list_functions_cursor(cli_runner, specs_dir):
    result = cli_runner.invoke(
        cli,
        ["--specs-dir", str(specs_dir), "doc", "list-functions", "--limit", "1"],
    )
    first = _load_output(result)
    cursor = first["meta"]["pagination"]["cursor"]
    names = [entry["name"] for entry in first["data"]["functions"]]
    assert cursor
    assert names == ["alpha"]

    second = cli_runner.invoke(
        cli,
        [
            "--specs-dir",
            str(specs_dir),
            "doc",
            "list-functions",
            "--limit",
            "1",
            "--cursor",
            cursor,
        ],
    )
    follow = _load_output(second)
    assert follow["data"]["functions"][0]["name"] == "beta"
    assert follow["data"]["offset"] == 1


def test_doc_search_cursor(cli_runner, specs_dir):
    result = cli_runner.invoke(
        cli,
        ["--specs-dir", str(specs_dir), "doc", "search", "alpha", "--limit", "1"],
    )
    first = _load_output(result)
    cursor = first["meta"]["pagination"]["cursor"]
    assert cursor
    assert first["data"]["matches"][0]["name"] == "alpha"

    second = cli_runner.invoke(
        cli,
        [
            "--specs-dir",
            str(specs_dir),
            "doc",
            "search",
            "alpha",
            "--limit",
            "1",
            "--cursor",
            cursor,
        ],
    )
    follow = _load_output(second)
    assert follow["data"]["matches"][0]["name"] == "beta"
    assert follow["data"]["offset"] == 1


def test_doc_refactor_candidates_cursor(cli_runner, specs_dir):
    result = cli_runner.invoke(
        cli,
        [
            "--specs-dir",
            str(specs_dir),
            "doc",
            "refactor-candidates",
            "--limit",
            "1",
        ],
    )
    first = _load_output(result)
    cursor = first["meta"]["pagination"]["cursor"]
    assert cursor
    assert first["data"]["candidates"][0]["name"] == "alpha"

    second = cli_runner.invoke(
        cli,
        [
            "--specs-dir",
            str(specs_dir),
            "doc",
            "refactor-candidates",
            "--limit",
            "1",
            "--cursor",
            cursor,
        ],
    )
    follow = _load_output(second)
    assert follow["data"]["candidates"][0]["name"] == "beta"
    assert follow["data"]["offset"] == 1


def test_doc_trace_calls_validates_max_depth(cli_runner, specs_dir):
    result = cli_runner.invoke(
        cli,
        [
            "--specs-dir",
            str(specs_dir),
            "doc",
            "trace-calls",
            "alpha",
            "--max-depth",
            "0",
        ],
    )
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["data"]["error_code"] == "VALIDATION_ERROR"


def test_doc_context_validates_depth(cli_runner, specs_dir):
    result = cli_runner.invoke(
        cli,
        [
            "--specs-dir",
            str(specs_dir),
            "doc",
            "context",
            "alpha",
            "--depth",
            "4",
        ],
    )
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["data"]["error_code"] == "VALIDATION_ERROR"


def test_doc_refactor_candidates_validates_threshold(cli_runner, specs_dir):
    result = cli_runner.invoke(
        cli,
        [
            "--specs-dir",
            str(specs_dir),
            "doc",
            "refactor-candidates",
            "--threshold",
            "-5",
        ],
    )
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["data"]["error_code"] == "VALIDATION_ERROR"
