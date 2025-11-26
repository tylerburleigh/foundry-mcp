"""
Base adapter interface for parity testing.

Defines the abstract interface that both foundry-mcp and sdd-toolkit
adapters must implement for comparison testing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class SpecToolAdapter(ABC):
    """
    Abstract adapter for spec operations.

    Both FoundryMcpAdapter and SddToolkitAdapter implement this interface,
    enabling the same test code to run against both systems.
    """

    def __init__(self, specs_dir: Path):
        """
        Initialize adapter.

        Args:
            specs_dir: Path to the specs directory containing
                       pending/, active/, completed/, archived/ folders
        """
        self.specs_dir = specs_dir

    # =========================================================================
    # Spec Operations
    # =========================================================================

    @abstractmethod
    def list_specs(self, status: str = "all") -> Dict[str, Any]:
        """
        List specifications by status.

        Args:
            status: Filter by status (all, pending, active, completed, archived)

        Returns:
            Dict with specs list and metadata
        """
        pass

    @abstractmethod
    def get_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Get specification details.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with spec data or error
        """
        pass

    @abstractmethod
    def get_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """
        Get task details.

        Args:
            spec_id: Specification identifier
            task_id: Task identifier

        Returns:
            Dict with task data or error
        """
        pass

    # =========================================================================
    # Task Operations
    # =========================================================================

    @abstractmethod
    def next_task(self, spec_id: str) -> Dict[str, Any]:
        """
        Find the next actionable task.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with next task info or message if none available
        """
        pass

    @abstractmethod
    def prepare_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """
        Prepare task context and dependencies.

        Args:
            spec_id: Specification identifier
            task_id: Task identifier

        Returns:
            Dict with task context, dependencies, and preparation info
        """
        pass

    @abstractmethod
    def check_dependencies(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """
        Check if task dependencies are satisfied.

        Args:
            spec_id: Specification identifier
            task_id: Task identifier

        Returns:
            Dict with dependency status (can_start, blockers, etc.)
        """
        pass

    @abstractmethod
    def update_status(
        self, spec_id: str, task_id: str, status: str
    ) -> Dict[str, Any]:
        """
        Update task status.

        Args:
            spec_id: Specification identifier
            task_id: Task identifier
            status: New status (pending, in_progress, completed, blocked)

        Returns:
            Dict with update result
        """
        pass

    @abstractmethod
    def complete_task(
        self, spec_id: str, task_id: str, journal_entry: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mark task as completed.

        Args:
            spec_id: Specification identifier
            task_id: Task identifier
            journal_entry: Optional completion note

        Returns:
            Dict with completion result
        """
        pass

    @abstractmethod
    def start_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """
        Mark task as in_progress.

        Args:
            spec_id: Specification identifier
            task_id: Task identifier

        Returns:
            Dict with start result
        """
        pass

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    @abstractmethod
    def progress(self, spec_id: str) -> Dict[str, Any]:
        """
        Get progress summary for a spec.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with progress percentage, counts, phase info
        """
        pass

    @abstractmethod
    def add_journal(
        self,
        spec_id: str,
        title: str,
        content: str,
        entry_type: str = "note",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a journal entry.

        Args:
            spec_id: Specification identifier
            title: Entry title
            content: Entry content
            entry_type: Type (note, decision, deviation, blocker)
            task_id: Optional associated task

        Returns:
            Dict with add result
        """
        pass

    @abstractmethod
    def get_journal(
        self, spec_id: str, entry_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get journal entries.

        Args:
            spec_id: Specification identifier
            entry_type: Optional filter by type

        Returns:
            Dict with journal entries
        """
        pass

    @abstractmethod
    def mark_blocked(
        self, spec_id: str, task_id: str, reason: str
    ) -> Dict[str, Any]:
        """
        Mark a task as blocked.

        Args:
            spec_id: Specification identifier
            task_id: Task identifier
            reason: Blocking reason

        Returns:
            Dict with block result
        """
        pass

    @abstractmethod
    def unblock(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """
        Remove blocked status from a task.

        Args:
            spec_id: Specification identifier
            task_id: Task identifier

        Returns:
            Dict with unblock result
        """
        pass

    @abstractmethod
    def list_blocked(self, spec_id: str) -> Dict[str, Any]:
        """
        List all blocked tasks.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with blocked tasks list
        """
        pass

    # =========================================================================
    # Validation
    # =========================================================================

    @abstractmethod
    def validate_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Validate specification structure.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with validation result (is_valid, errors, warnings)
        """
        pass

    @abstractmethod
    def fix_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Auto-fix spec issues.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with fix results (fixes_applied, remaining_issues)
        """
        pass

    @abstractmethod
    def spec_stats(self, spec_id: str) -> Dict[str, Any]:
        """
        Get specification statistics.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with stats (task counts, completion %, depth, etc.)
        """
        pass

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @abstractmethod
    def activate_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Move spec from pending to active.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with activation result
        """
        pass

    @abstractmethod
    def complete_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Move spec from active to completed.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with completion result
        """
        pass

    @abstractmethod
    def archive_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Move spec to archived.

        Args:
            spec_id: Specification identifier

        Returns:
            Dict with archive result
        """
        pass

    @abstractmethod
    def move_spec(self, spec_id: str, target_status: str) -> Dict[str, Any]:
        """
        Move spec to a specific status folder.

        Args:
            spec_id: Specification identifier
            target_status: Target folder (pending, active, completed, archived)

        Returns:
            Dict with move result
        """
        pass
