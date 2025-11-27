"""
Environment tools for foundry-mcp.

Provides MCP tools for environment verification, workspace initialization,
and topology detection.
"""

import logging
import shutil
import subprocess
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool

logger = logging.getLogger(__name__)


def register_environment_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register environment tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="sdd-verify-toolchain",
    )
    def sdd_verify_toolchain(
        include_optional: bool = True,
    ) -> dict:
        """
        Verify local CLI and toolchain availability.

        Performs a sanity check of required and optional binaries needed
        for SDD workflows. Returns availability status for each tool.

        WHEN TO USE:
        - Before starting a new SDD workflow
        - Diagnosing environment issues
        - Validating CI/CD environment setup
        - Troubleshooting missing dependencies

        Args:
            include_optional: Include optional tools in check (default: True)

        Returns:
            JSON object with tool availability status:
            - required: Dict of required tools and their availability
            - optional: Dict of optional tools and their availability (if requested)
            - all_available: Boolean indicating if all required tools are present
            - missing: List of missing required tools (if any)
        """
        try:
            # Define required and optional tools
            required_tools = ["python", "git"]
            optional_tools = ["grep", "cat", "find", "node", "npm"]

            def check_tool(tool_name: str) -> bool:
                """Check if a tool is available in PATH."""
                return shutil.which(tool_name) is not None

            # Check required tools
            required_status: Dict[str, bool] = {}
            missing_required: List[str] = []
            for tool in required_tools:
                available = check_tool(tool)
                required_status[tool] = available
                if not available:
                    missing_required.append(tool)

            # Check optional tools if requested
            optional_status: Dict[str, bool] = {}
            if include_optional:
                for tool in optional_tools:
                    optional_status[tool] = check_tool(tool)

            # Build response data
            all_available = len(missing_required) == 0
            data: Dict[str, Any] = {
                "required": required_status,
                "all_available": all_available,
            }

            if include_optional:
                data["optional"] = optional_status

            if missing_required:
                data["missing"] = missing_required

            # Add warnings for missing optional tools
            warnings: List[str] = []
            if include_optional:
                missing_optional = [
                    tool for tool, available in optional_status.items() if not available
                ]
                if missing_optional:
                    warnings.append(
                        f"Optional tools not found: {', '.join(missing_optional)}"
                    )

            if not all_available:
                return asdict(
                    error_response(
                        f"Required tools missing: {', '.join(missing_required)}",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        data=data,
                        remediation="Install missing tools before proceeding with SDD workflows.",
                    )
                )

            return asdict(
                success_response(
                    data=data,
                    warnings=warnings if warnings else None,
                )
            )

        except Exception as e:
            logger.exception("Error verifying toolchain")
            return asdict(error_response(f"Failed to verify toolchain: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="sdd-init-workspace",
    )
    def sdd_init_workspace(
        path: Optional[str] = None,
        create_subdirs: bool = True,
    ) -> dict:
        """
        Bootstrap working directory for SDD workflows.

        Initializes the specs/ directory structure needed for SDD workflows.
        Creates the standard folder hierarchy (active, pending, completed, archived).

        WHEN TO USE:
        - Setting up a new project for SDD
        - Onboarding an existing repository to SDD
        - Recreating missing directory structure

        Args:
            path: Project root path (default: current directory)
            create_subdirs: Create standard subdirectories (default: True)

        Returns:
            JSON object with initialization status:
            - success: Boolean indicating if initialization succeeded
            - specs_dir: Path to the specs directory
            - created_dirs: List of directories that were created
            - existing_dirs: List of directories that already existed
        """
        import os
        from pathlib import Path

        try:
            # Determine base path
            base_path = Path(path) if path else Path.cwd()
            specs_dir = base_path / "specs"

            # Standard subdirectories
            subdirs = ["active", "pending", "completed", "archived"]

            created_dirs: List[str] = []
            existing_dirs: List[str] = []

            # Create specs directory if needed
            if not specs_dir.exists():
                specs_dir.mkdir(parents=True)
                created_dirs.append(str(specs_dir))
            else:
                existing_dirs.append(str(specs_dir))

            # Create subdirectories if requested
            if create_subdirs:
                for subdir in subdirs:
                    subdir_path = specs_dir / subdir
                    if not subdir_path.exists():
                        subdir_path.mkdir(parents=True)
                        created_dirs.append(str(subdir_path))
                    else:
                        existing_dirs.append(str(subdir_path))

            data: Dict[str, Any] = {
                "specs_dir": str(specs_dir),
                "active_dir": str(specs_dir / "active"),
            }

            if created_dirs:
                data["created_dirs"] = created_dirs
            if existing_dirs:
                data["existing_dirs"] = existing_dirs

            # Add warning if nothing was created
            warnings: List[str] = []
            if not created_dirs:
                warnings.append("All directories already existed, no changes made")

            return asdict(
                success_response(
                    data=data,
                    warnings=warnings if warnings else None,
                )
            )

        except PermissionError as e:
            logger.exception("Permission denied during workspace initialization")
            return asdict(
                error_response(
                    f"Permission denied: {e}",
                    error_code="FORBIDDEN",
                    error_type="authorization",
                    remediation="Check write permissions for the target directory.",
                )
            )
        except Exception as e:
            logger.exception("Error initializing workspace")
            return asdict(error_response(f"Failed to initialize workspace: {e}"))
