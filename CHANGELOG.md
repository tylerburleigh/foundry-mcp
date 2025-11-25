# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Task operations module (`foundry_mcp.core.task`):
  - `get_next_task` - Find next actionable task based on status and dependencies
  - `check_dependencies` - Analyze blocking and soft dependencies
  - `prepare_task` - Prepare complete context for task implementation
  - `get_previous_sibling` - Get context from previously executed sibling task
  - `get_parent_context` - Get parent task metadata and position
  - `get_phase_context` - Get current phase progress and blockers
  - `get_task_journal_summary` - Get journal entries for a task
  - `is_unblocked` / `is_in_current_phase` - Dependency and phase helpers
- Progress calculation module (`foundry_mcp.core.progress`):
  - `recalculate_progress` - Recursively update task counts
  - `update_parent_status` - Propagate status changes up hierarchy
  - `get_progress_summary` - Get completion percentages and stats
  - `list_phases` - List all phases with progress
  - `get_task_counts_by_status` - Count tasks by status
- MCP task tools (`foundry_mcp.tools.tasks`):
  - `foundry_prepare_task` - Prepare task with full context
  - `foundry_next_task` - Find next actionable task
  - `foundry_task_info` - Get detailed task information
  - `foundry_check_deps` - Check dependency status
  - `foundry_update_status` - Update task status
  - `foundry_complete_task` - Mark task complete with journal
  - `foundry_start_task` - Start working on a task
  - `foundry_progress` - Get spec/phase progress
- Unit tests for task operations (30 tests)

## [0.1.0] - 2025-01-25

### Added
- Initial project setup with pyproject.toml and hatchling build system
- Core spec operations module (`foundry_mcp.core.spec`):
  - `load_spec` - Load JSON spec files by ID or path
  - `save_spec` - Save specs with atomic writes and automatic backups
  - `find_spec_file` - Locate spec files across status folders
  - `find_specs_directory` - Auto-discover specs directory
  - `list_specs` - List specs with filtering by status
  - `get_node` / `update_node` - Hierarchy node operations
- Package structure following Python best practices
- FastMCP and MCP dependencies for server implementation

### Technical Decisions
- Extracted core spec operations from claude-sdd-toolkit as standalone module
- Removed external dependencies for portability
- Atomic file writes with `.tmp` extension for data safety
- Automatic backup creation in `.backups/` directory before saves
