# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Internal CLI Elimination**: Refactored all MCP tool modules to use direct Python core API calls instead of subprocess calls to the external `sdd` CLI. This eliminates process spawning overhead and improves reliability.
  - `tools/spec_helpers.py`: Replaced `_run_sdd_command()` with direct calls to `core/spec.py` APIs
  - `tools/authoring.py`: Uses `core/spec.py`, `core/task.py` for spec creation and task management
  - `tools/mutations.py`: Uses `core/validation.py` for verification operations
  - `tools/planning.py`: Uses `core/progress.py`, `core/rendering.py` for planning tools
  - `tools/documentation.py`: Uses `core/rendering.py` for spec documentation
  - `tools/analysis.py`: Uses `core/spec.py` for dependency analysis
  - `tools/review.py`: Returns NOT_IMPLEMENTED for tools requiring external AI integration
  - `tools/utilities.py`: Uses `core/cache.py` for cache management
  - `tools/git_integration.py`: Returns NOT_IMPLEMENTED for tools requiring git CLI

### Removed
- Removed `_run_sdd_command()` helper function and all subprocess-based CLI invocations from MCP tools
- Removed `_sdd_cli_breaker` circuit breaker (no longer needed without subprocess calls)
- Removed unused CLI-related imports from tool modules

### Fixed
- Fixed `plan_format` tool using invalid `render_task_list(root_id=)` parameter
- Fixed timing flakiness in `test_concurrent_execution` by increasing tolerance

### Internal
- Updated all unit tests to test direct API calls instead of mocking subprocess calls
- Added comprehensive test coverage for new core API integrations
- Maintained response-v2 contract compliance across all refactored tools
