"""
Unit tests for foundry_mcp.core.rendering module.

Tests rendering functions for spec-to-markdown conversion.
"""

import pytest
from foundry_mcp.core.rendering import (
    render_spec_to_markdown,
    render_progress_bar,
    render_task_list,
    get_status_icon,
    RenderOptions,
    RenderResult,
    STATUS_ICONS,
)


# Test fixtures

@pytest.fixture
def valid_spec():
    """Return a minimal valid spec for testing rendering."""
    return {
        "spec_id": "test-spec-2025-01-01-001",
        "metadata": {
            "title": "Test Specification",
            "description": "A test spec for rendering",
            "objectives": ["Objective 1", "Objective 2"],
            "estimated_hours": 10,
            "complexity": "medium",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "in_progress",
                "parent": None,
                "children": ["phase-1"],
                "total_tasks": 3,
                "completed_tasks": 1,
                "metadata": {},
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase One",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["group-1"],
                "total_tasks": 3,
                "completed_tasks": 1,
                "metadata": {
                    "purpose": "First phase purpose",
                    "risk_level": "low",
                    "estimated_hours": 5,
                },
                "dependencies": {
                    "blocked_by": [],
                    "depends": [],
                },
            },
            "group-1": {
                "type": "group",
                "title": "Implementation Tasks",
                "status": "in_progress",
                "parent": "phase-1",
                "children": ["task-1", "task-2", "verify-1"],
                "total_tasks": 3,
                "completed_tasks": 1,
                "metadata": {},
                "dependencies": {
                    "blocked_by": [],
                },
            },
            "task-1": {
                "type": "task",
                "title": "First Task",
                "status": "completed",
                "parent": "group-1",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 1,
                "metadata": {
                    "file_path": "src/module.py",
                    "estimated_hours": 2,
                    "changes": "Add new function",
                    "reasoning": "Needed for feature X",
                },
                "dependencies": {
                    "depends": [],
                    "blocked_by": [],
                },
            },
            "task-2": {
                "type": "task",
                "title": "Second Task",
                "status": "pending",
                "parent": "group-1",
                "children": ["subtask-1"],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {
                    "file_path": "src/other.py",
                    "details": ["Detail 1", "Detail 2"],
                },
                "dependencies": {
                    "depends": ["task-1"],
                    "blocked_by": [],
                },
            },
            "subtask-1": {
                "type": "subtask",
                "title": "Subtask for Task 2",
                "status": "pending",
                "parent": "task-2",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
                "dependencies": {},
            },
            "verify-1": {
                "type": "verify",
                "title": "Verify Implementation",
                "status": "pending",
                "parent": "group-1",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {
                    "verification_type": "auto",
                    "command": "pytest tests/",
                    "expected": "All tests pass",
                },
                "dependencies": {},
            },
        },
        "journal": [
            {
                "timestamp": "2025-01-01T12:00:00Z",
                "entry_type": "status_change",
                "title": "Task started",
                "task_id": "task-1",
            },
            {
                "timestamp": "2025-01-02T10:00:00Z",
                "entry_type": "decision",
                "title": "Architectural decision",
                "task_id": "task-1",
            },
        ],
    }


class TestRenderProgressBar:
    """Tests for render_progress_bar function."""

    def test_zero_progress(self):
        """Test progress bar with 0 completion."""
        bar = render_progress_bar(0, 10, width=10)
        assert bar == "[░░░░░░░░░░] 0%"

    def test_full_progress(self):
        """Test progress bar with 100% completion."""
        bar = render_progress_bar(10, 10, width=10)
        assert bar == "[██████████] 100%"

    def test_half_progress(self):
        """Test progress bar with 50% completion."""
        bar = render_progress_bar(5, 10, width=10)
        assert bar == "[█████░░░░░] 50%"

    def test_empty_total(self):
        """Test progress bar with zero total."""
        bar = render_progress_bar(0, 0, width=10)
        assert bar == "[░░░░░░░░░░] 0%"

    def test_custom_width(self):
        """Test progress bar with custom width."""
        bar = render_progress_bar(2, 4, width=20)
        assert bar == "[██████████░░░░░░░░░░] 50%"
        assert len(bar.split("]")[0]) == 21  # 1 for '[' + 20 for bar


class TestGetStatusIcon:
    """Tests for get_status_icon function."""

    def test_pending_status(self):
        """Test icon for pending status."""
        assert get_status_icon("pending") == "⏳"

    def test_in_progress_status(self):
        """Test icon for in_progress status."""
        assert get_status_icon("in_progress") == "🔄"

    def test_completed_status(self):
        """Test icon for completed status."""
        assert get_status_icon("completed") == "✅"

    def test_blocked_status(self):
        """Test icon for blocked status."""
        assert get_status_icon("blocked") == "🚫"

    def test_failed_status(self):
        """Test icon for failed status."""
        assert get_status_icon("failed") == "❌"

    def test_unknown_status(self):
        """Test icon for unknown status."""
        assert get_status_icon("unknown_status") == "❓"


class TestStatusIcons:
    """Tests for STATUS_ICONS constant."""

    def test_all_statuses_defined(self):
        """Test that all expected statuses have icons."""
        expected = ["pending", "in_progress", "completed", "blocked", "failed"]
        for status in expected:
            assert status in STATUS_ICONS


class TestRenderSpecToMarkdown:
    """Tests for render_spec_to_markdown function."""

    def test_returns_render_result(self, valid_spec):
        """Test that function returns RenderResult."""
        result = render_spec_to_markdown(valid_spec)
        assert isinstance(result, RenderResult)

    def test_result_contains_markdown(self, valid_spec):
        """Test that result contains markdown content."""
        result = render_spec_to_markdown(valid_spec)
        assert result.markdown
        assert isinstance(result.markdown, str)

    def test_result_contains_metadata(self, valid_spec):
        """Test that result contains spec metadata."""
        result = render_spec_to_markdown(valid_spec)
        assert result.spec_id == "test-spec-2025-01-01-001"
        assert result.title == "Test Specification"

    def test_result_contains_counts(self, valid_spec):
        """Test that result contains task counts."""
        result = render_spec_to_markdown(valid_spec)
        assert result.total_tasks == 3
        assert result.completed_tasks == 1
        assert result.total_sections == 1  # One phase

    def test_markdown_contains_title(self, valid_spec):
        """Test that markdown contains spec title."""
        result = render_spec_to_markdown(valid_spec)
        assert "# Test Specification" in result.markdown

    def test_markdown_contains_spec_id(self, valid_spec):
        """Test that markdown contains spec ID."""
        result = render_spec_to_markdown(valid_spec)
        assert "test-spec-2025-01-01-001" in result.markdown

    def test_markdown_contains_progress_bar(self, valid_spec):
        """Test that markdown contains progress bar."""
        result = render_spec_to_markdown(valid_spec)
        assert "█" in result.markdown or "░" in result.markdown

    def test_markdown_contains_objectives(self, valid_spec):
        """Test that markdown contains objectives."""
        result = render_spec_to_markdown(valid_spec)
        assert "Objective 1" in result.markdown
        assert "Objective 2" in result.markdown

    def test_markdown_contains_phases(self, valid_spec):
        """Test that markdown contains phase titles."""
        result = render_spec_to_markdown(valid_spec)
        assert "Phase One" in result.markdown

    def test_markdown_contains_tasks(self, valid_spec):
        """Test that markdown contains task titles."""
        result = render_spec_to_markdown(valid_spec)
        assert "First Task" in result.markdown
        assert "Second Task" in result.markdown

    def test_markdown_contains_verification(self, valid_spec):
        """Test that markdown contains verification tasks."""
        result = render_spec_to_markdown(valid_spec)
        assert "Verify Implementation" in result.markdown

    def test_default_options_applied(self, valid_spec):
        """Test that default options are applied."""
        result = render_spec_to_markdown(valid_spec)
        # Default should include metadata and progress
        assert "Spec ID" in result.markdown
        assert "Progress" in result.markdown

    def test_custom_options_respected(self, valid_spec):
        """Test that custom options are respected."""
        options = RenderOptions(
            include_metadata=False,
            include_progress=False,
        )
        result = render_spec_to_markdown(valid_spec, options)
        # Should still have title but not detailed metadata
        assert "# Test Specification" in result.markdown

    def test_include_journal_option(self, valid_spec):
        """Test that journal inclusion option works."""
        options = RenderOptions(include_journal=True)
        result = render_spec_to_markdown(valid_spec, options)
        assert "Journal" in result.markdown or "status_change" in result.markdown

    def test_phase_filter_option(self, valid_spec):
        """Test that phase filter option works."""
        # Add another phase
        valid_spec["hierarchy"]["spec-root"]["children"].append("phase-2")
        valid_spec["hierarchy"]["phase-2"] = {
            "type": "phase",
            "title": "Phase Two",
            "status": "pending",
            "parent": "spec-root",
            "children": [],
            "total_tasks": 0,
            "completed_tasks": 0,
            "metadata": {},
        }

        options = RenderOptions(phase_filter=["phase-1"])
        result = render_spec_to_markdown(valid_spec, options)
        assert "Phase One" in result.markdown
        assert "Phase Two" not in result.markdown

    def test_max_depth_option(self, valid_spec):
        """Test that max_depth option limits rendering depth."""
        options = RenderOptions(max_depth=1)
        result = render_spec_to_markdown(valid_spec, options)
        # With depth 1, should show phases but limited groups/tasks
        assert "Phase One" in result.markdown

    def test_handles_empty_spec(self):
        """Test handling of minimal/empty spec."""
        empty_spec = {
            "spec_id": "empty-spec-001",
            "metadata": {},
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Empty Spec",
                    "status": "pending",
                    "parent": None,
                    "children": [],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {},
                }
            },
        }
        result = render_spec_to_markdown(empty_spec)
        assert result.markdown
        assert result.total_sections == 0

    def test_handles_missing_metadata(self):
        """Test handling of spec without metadata."""
        spec = {
            "spec_id": "no-meta-001",
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "No Metadata Spec",
                    "status": "pending",
                    "parent": None,
                    "children": [],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {},
                }
            },
        }
        result = render_spec_to_markdown(spec)
        assert result.markdown
        assert "No Metadata Spec" in result.markdown


class TestRenderTaskList:
    """Tests for render_task_list function."""

    def test_returns_markdown_list(self, valid_spec):
        """Test that function returns markdown list."""
        result = render_task_list(valid_spec)
        assert isinstance(result, str)
        assert "## Task List" in result

    def test_includes_tasks(self, valid_spec):
        """Test that list includes tasks."""
        result = render_task_list(valid_spec)
        assert "First Task" in result
        assert "Second Task" in result

    def test_includes_subtasks(self, valid_spec):
        """Test that list includes subtasks."""
        result = render_task_list(valid_spec)
        assert "Subtask for Task 2" in result

    def test_includes_verification(self, valid_spec):
        """Test that list includes verification tasks."""
        result = render_task_list(valid_spec)
        assert "Verify Implementation" in result

    def test_status_filter(self, valid_spec):
        """Test that status filter works."""
        result = render_task_list(valid_spec, status_filter="completed")
        assert "First Task" in result
        assert "Second Task" not in result

    def test_exclude_completed(self, valid_spec):
        """Test that completed tasks can be excluded."""
        result = render_task_list(valid_spec, include_completed=False)
        assert "First Task" not in result
        assert "Second Task" in result

    def test_includes_file_paths(self, valid_spec):
        """Test that file paths are included."""
        result = render_task_list(valid_spec)
        assert "src/module.py" in result

    def test_includes_status_icons(self, valid_spec):
        """Test that status icons are included."""
        result = render_task_list(valid_spec)
        assert "✅" in result  # Completed task icon

    def test_empty_list_message(self):
        """Test message when no tasks match."""
        spec = {
            "spec_id": "no-tasks-001",
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "No Tasks",
                    "status": "pending",
                    "children": [],
                }
            },
        }
        result = render_task_list(spec)
        assert "No tasks found" in result


class TestRenderOptions:
    """Tests for RenderOptions dataclass."""

    def test_default_values(self):
        """Test default option values."""
        options = RenderOptions()
        assert options.mode == "basic"
        assert options.include_metadata is True
        assert options.include_progress is True
        assert options.include_dependencies is True
        assert options.include_journal is False
        assert options.max_depth == 0
        assert options.phase_filter is None

    def test_custom_values(self):
        """Test custom option values."""
        options = RenderOptions(
            mode="enhanced",
            include_metadata=False,
            include_progress=False,
            include_dependencies=False,
            include_journal=True,
            max_depth=2,
            phase_filter=["phase-1", "phase-2"],
        )
        assert options.mode == "enhanced"
        assert options.include_metadata is False
        assert options.include_progress is False
        assert options.include_dependencies is False
        assert options.include_journal is True
        assert options.max_depth == 2
        assert options.phase_filter == ["phase-1", "phase-2"]


class TestRenderResult:
    """Tests for RenderResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = RenderResult(
            markdown="# Test",
            spec_id="test-001",
            title="Test",
        )
        assert result.markdown == "# Test"
        assert result.spec_id == "test-001"
        assert result.title == "Test"
        assert result.total_sections == 0
        assert result.total_tasks == 0
        assert result.completed_tasks == 0

    def test_custom_values(self):
        """Test custom result values."""
        result = RenderResult(
            markdown="# Test\n\nContent",
            spec_id="test-001",
            title="Test Spec",
            total_sections=3,
            total_tasks=10,
            completed_tasks=5,
        )
        assert result.total_sections == 3
        assert result.total_tasks == 10
        assert result.completed_tasks == 5


class TestMarkdownValidity:
    """Tests to verify rendered markdown is valid."""

    def test_headers_are_valid(self, valid_spec):
        """Test that markdown headers are valid."""
        result = render_spec_to_markdown(valid_spec)
        lines = result.markdown.split("\n")
        for line in lines:
            if line.startswith("#"):
                # Headers should have space after #
                header_level = 0
                for char in line:
                    if char == "#":
                        header_level += 1
                    else:
                        break
                # After the # chars, should be a space
                if header_level > 0 and len(line) > header_level:
                    assert line[header_level] == " ", f"Invalid header: {line}"

    def test_list_items_are_valid(self, valid_spec):
        """Test that list items are valid markdown."""
        result = render_task_list(valid_spec)
        lines = result.split("\n")
        for line in lines:
            if line.startswith("-"):
                # List items should have space after -
                assert line[1] == " ", f"Invalid list item: {line}"

    def test_bold_text_is_paired(self, valid_spec):
        """Test that bold markers are properly paired."""
        result = render_spec_to_markdown(valid_spec)
        # Count ** markers - should be even
        count = result.markdown.count("**")
        assert count % 2 == 0, "Unpaired bold markers"

    def test_code_blocks_are_paired(self, valid_spec):
        """Test that code block markers are paired."""
        result = render_spec_to_markdown(valid_spec)
        # Count ``` markers - should be even
        count = result.markdown.count("```")
        assert count % 2 == 0, "Unpaired code block markers"
