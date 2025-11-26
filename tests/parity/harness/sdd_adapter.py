"""
SDD-Toolkit CLI adapter for parity testing.

Invokes sdd CLI commands via subprocess and parses JSON output.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import SpecToolAdapter


class SddToolkitAdapter(SpecToolAdapter):
    """
    Adapter for sdd-toolkit CLI.

    Invokes CLI commands via subprocess with --json output flag.
    """

    def __init__(self, specs_dir: Path):
        """
        Initialize adapter.

        Args:
            specs_dir: Path to the specs directory
        """
        super().__init__(specs_dir)
        # Project root is parent of specs/
        self.project_root = specs_dir.parent

    def _run_sdd(
        self,
        *args: str,
        timeout: int = 30,
        check: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute sdd CLI command and return parsed JSON output.

        Args:
            *args: Command arguments (e.g., "list-specs", "--status", "active")
            timeout: Command timeout in seconds
            check: If True, raise on non-zero exit

        Returns:
            Parsed JSON response or error dict
        """
        cmd = ["sdd", "--json", "--path", str(self.project_root)]
        cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root),
            )

            # Try to parse JSON from stdout
            if result.stdout.strip():
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return raw output
                    pass

            # Check for error
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr or f"Command failed with exit code {result.returncode}",
                    "_exit_code": result.returncode,
                    "_stdout": result.stdout,
                }

            # No JSON output but command succeeded
            return {
                "success": True,
                "_stdout": result.stdout,
                "_stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "sdd command not found. Is sdd-toolkit installed?",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # Spec Operations
    # =========================================================================

    def list_specs(self, status: str = "all") -> Dict[str, Any]:
        """List specifications by status."""
        args = ["list-specs"]
        if status != "all":
            args.extend(["--status", status])
        return self._run_sdd(*args)

    def get_spec(self, spec_id: str) -> Dict[str, Any]:
        """Get specification details."""
        # sdd-toolkit uses 'progress' command for spec overview
        return self._run_sdd("progress", spec_id)

    def get_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Get task details."""
        return self._run_sdd("task-info", spec_id, task_id)

    # =========================================================================
    # Task Operations
    # =========================================================================

    def next_task(self, spec_id: str) -> Dict[str, Any]:
        """Find the next actionable task."""
        return self._run_sdd("next-task", spec_id)

    def prepare_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Prepare task context and dependencies."""
        return self._run_sdd("prepare-task", spec_id, task_id)

    def check_dependencies(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Check if task dependencies are satisfied."""
        return self._run_sdd("check-deps", spec_id, task_id)

    def update_status(
        self, spec_id: str, task_id: str, status: str
    ) -> Dict[str, Any]:
        """Update task status."""
        return self._run_sdd("update-status", spec_id, task_id, status)

    def complete_task(
        self, spec_id: str, task_id: str, journal_entry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark task as completed."""
        args = ["complete-task", spec_id, task_id]
        if journal_entry:
            args.extend(["--message", journal_entry])
        return self._run_sdd(*args)

    def start_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Mark task as in_progress."""
        return self.update_status(spec_id, task_id, "in_progress")

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def progress(self, spec_id: str) -> Dict[str, Any]:
        """Get progress summary for a spec."""
        return self._run_sdd("progress", spec_id)

    def add_journal(
        self,
        spec_id: str,
        title: str,
        content: str,
        entry_type: str = "note",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a journal entry."""
        args = ["add-journal", spec_id, "--title", title, "--content", content]
        if entry_type != "note":
            args.extend(["--type", entry_type])
        if task_id:
            args.extend(["--task", task_id])
        return self._run_sdd(*args)

    def get_journal(
        self, spec_id: str, entry_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get journal entries."""
        args = ["get-journal", spec_id]
        if entry_type:
            args.extend(["--type", entry_type])
        return self._run_sdd(*args)

    def mark_blocked(
        self, spec_id: str, task_id: str, reason: str
    ) -> Dict[str, Any]:
        """Mark a task as blocked."""
        return self._run_sdd("mark-blocked", spec_id, task_id, "--reason", reason)

    def unblock(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Remove blocked status from a task."""
        return self._run_sdd("unblock-task", spec_id, task_id)

    def list_blocked(self, spec_id: str) -> Dict[str, Any]:
        """List all blocked tasks."""
        return self._run_sdd("list-blockers", spec_id)

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_spec(self, spec_id: str) -> Dict[str, Any]:
        """Validate specification structure."""
        return self._run_sdd("validate", spec_id)

    def fix_spec(self, spec_id: str) -> Dict[str, Any]:
        """Auto-fix spec issues."""
        return self._run_sdd("validate", spec_id, "--fix")

    def spec_stats(self, spec_id: str) -> Dict[str, Any]:
        """Get specification statistics."""
        # sdd-toolkit has 'stats' or 'status-report' command
        return self._run_sdd("stats", spec_id)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def activate_spec(self, spec_id: str) -> Dict[str, Any]:
        """Move spec from pending to active."""
        return self._run_sdd("activate-spec", spec_id)

    def complete_spec(self, spec_id: str) -> Dict[str, Any]:
        """Move spec from active to completed."""
        return self._run_sdd("complete-spec", spec_id)

    def archive_spec(self, spec_id: str) -> Dict[str, Any]:
        """Move spec to archived."""
        return self._run_sdd("move-spec", spec_id, "archived")

    def move_spec(self, spec_id: str, target_status: str) -> Dict[str, Any]:
        """Move spec to a specific status folder."""
        return self._run_sdd("move-spec", spec_id, target_status)
