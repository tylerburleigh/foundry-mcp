"""Tests for the native `sdd doc scope` command."""

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from foundry_mcp.cli.main import cli


class StubDocsQuery:
    """Deterministic DocsQuery stub used for CLI tests."""

    should_load = True
    scope_success = True

    def __init__(self, workspace=None):
        self.workspace = workspace

    def load(self) -> bool:  # pragma: no cover - trivial stub
        return self.__class__.should_load

    def get_scope(self, target: str, mode: str = "plan") -> SimpleNamespace:
        if not self.__class__.scope_success:
            return SimpleNamespace(success=False, error="scope failed", results=[])

        scope = {
            "file_path": target,
            "mode": mode,
            "classes": [
                {"name": "Alpha", "line": 10, "bases": []},
            ],
            "functions": [
                {"name": "process_data", "line": 20, "signature": "process_data()"},
            ],
        }
        return SimpleNamespace(success=True, results=[scope])

    def trace_calls(
        self, function_name: str, direction: str = "both", max_depth: int = 2
    ) -> SimpleNamespace:  # noqa: D401,E501
        entries = [
            SimpleNamespace(caller=f"caller_{function_name}", callee=function_name),
        ]
        return SimpleNamespace(success=True, results=entries)

    def find_classes_in_file(
        self, file_path: str
    ) -> SimpleNamespace:  # pragma: no cover - unused helper
        return SimpleNamespace(results=[])

    def find_functions_in_file(
        self, file_path: str
    ) -> SimpleNamespace:  # pragma: no cover - unused helper
        return SimpleNamespace(results=[])


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def patch_docs_query(monkeypatch):
    # Reset stub flags before each test
    StubDocsQuery.should_load = True
    StubDocsQuery.scope_success = True
    monkeypatch.setattr("foundry_mcp.core.docs.DocsQuery", StubDocsQuery)


@pytest.fixture
def specs_dir(tmp_path):
    specs = tmp_path / "specs"
    specs.mkdir()
    return specs


def _invoke_scope(cli_runner: CliRunner, specs_dir, *args):
    return cli_runner.invoke(
        cli,
        ["--specs-dir", str(specs_dir), "doc", "scope", *args],
    )


def test_doc_scope_plan_view_returns_scope(cli_runner, specs_dir):
    result = _invoke_scope(cli_runner, specs_dir, "src/module.py", "--view", "plan")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["data"]["scope"]["functions"][0]["name"] == "process_data"


def test_doc_scope_trace_view_includes_trace(cli_runner, specs_dir):
    result = _invoke_scope(cli_runner, specs_dir, "src/module.py", "--view", "trace")
    assert result.exit_code == 0, result.output
    scope = json.loads(result.output)["data"]["scope"]
    assert "trace" in scope
    assert scope["trace"][0]["calls"][0]["callee"] == "process_data"


def test_doc_scope_load_failure_returns_error(cli_runner, specs_dir):
    StubDocsQuery.should_load = False
    result = _invoke_scope(cli_runner, specs_dir, "src/module.py")
    assert result.exit_code != 0
    assert "DOCS_NOT_FOUND" in result.output


def test_doc_scope_scope_failure_returns_error(cli_runner, specs_dir):
    StubDocsQuery.scope_success = False
    result = _invoke_scope(cli_runner, specs_dir, "src/module.py")
    assert result.exit_code != 0
    assert "scope failed" in result.output
