# 4. Testing & Parity

> Prove the native CLI behaves exactly like the MCP adapters (and any legacy CLIs we still reference).

## 4.1 Unit Coverage

- Add unit tests for every command module under `tests/unit/test_sdd_cli_*.py`.
- Mock shared services sparingly—favor lightweight fixture data that exercises serialization helpers end-to-end.

## 4.2 Integration & Golden Tests

- Use integration tests to exercise real filesystem/spec fixtures (see [docs/mcp_best_practices/10-testing-fixtures.md](../mcp_best_practices/10-testing-fixtures.md)).
- Capture golden outputs for complex commands and compare against expected results to catch regressions.

## 4.3 Parity Gates

- Maintain a dedicated parity suite (e.g., `pytest tests/integration/test_sdd_cli_parity.py`) that diff-checks canonical commands across runtimes.
- Run the parity suite in CI whenever CLI or MCP tool code changes. Break-glass by adding TODOs to the relevant spec if parity must be temporarily skipped.

## 4.4 Documentation & Reporting

- Document new test commands or fixtures inside the relevant spec nodes so future maintainers know how to regenerate them.
- Surface pass/fail status back into the spec verification nodes (`verify-*`) to keep the SDD workflow honest.
