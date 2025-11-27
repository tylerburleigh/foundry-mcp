"""
Mutation tools for foundry-mcp.

Provides MCP tools for batch modifications to SDD specifications.
These tools wrap SDD CLI commands for applying plans, verification,
and metadata mutations.

Resilience features:
- Circuit breaker for SDD CLI calls (opens after 5 consecutive failures)
- Timing metrics for all tool invocations
- Configurable timeout (default 30s per operation)
"""

import json
import logging
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    MEDIUM_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Metrics singleton for mutation tools
_metrics = get_metrics()

# Circuit breaker for SDD CLI operations
# Opens after 5 consecutive failures, recovers after 30 seconds
_sdd_cli_breaker = CircuitBreaker(
    name="sdd_cli_mutations",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3,
)

# Default timeout for CLI operations (30 seconds)
CLI_TIMEOUT: float = MEDIUM_TIMEOUT


def _run_sdd_command(
    cmd: List[str],
    tool_name: str,
    timeout: float = CLI_TIMEOUT,
) -> subprocess.CompletedProcess:
    """
    Execute an SDD CLI command with circuit breaker protection and timing.

    Args:
        cmd: Command list to execute
        tool_name: Name of the calling tool (for metrics)
        timeout: Timeout in seconds

    Returns:
        CompletedProcess result from subprocess.run

    Raises:
        CircuitBreakerError: If circuit breaker is open
        subprocess.TimeoutExpired: If command times out
        FileNotFoundError: If SDD CLI is not found
    """
    # Check circuit breaker
    if not _sdd_cli_breaker.can_execute():
        status = _sdd_cli_breaker.get_status()
        _metrics.counter(f"mutations.{tool_name}", labels={"status": "circuit_open"})
        raise CircuitBreakerError(
            f"SDD CLI circuit breaker is open (retry after {status.get('retry_after_seconds', 0):.1f}s)",
            breaker_name="sdd_cli_mutations",
            state=_sdd_cli_breaker.state,
            retry_after=status.get("retry_after_seconds"),
        )

    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Record success or failure based on return code
        if result.returncode == 0:
            _sdd_cli_breaker.record_success()
        else:
            # Non-zero return code counts as a failure for circuit breaker
            _sdd_cli_breaker.record_failure()

        return result

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        # These are infrastructure failures that should trip the circuit breaker
        _sdd_cli_breaker.record_failure()
        raise
    finally:
        # Record timing metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.timer(f"mutations.{tool_name}.duration_ms", elapsed_ms)


def register_mutation_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register mutation tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-apply-plan",
    )
    def spec_apply_plan(
        spec_id: str,
        modifications_file: str,
        dry_run: bool = False,
        output_file: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Apply bulk structural edits from a diff/plan JSON file.

        Wraps the SDD CLI apply-modifications command to batch apply changes
        to a specification from a JSON file. The modifications file contains
        structured changes like task additions, updates, removals, and
        metadata modifications.

        WHEN TO USE:
        - Applying review feedback to a specification
        - Batch updating multiple tasks from parsed review output
        - Migrating specs with automated transformations
        - Applying AI-generated spec modifications
        - Implementing bulk changes from sdd parse-review output

        Args:
            spec_id: Specification ID to modify
            modifications_file: Path to JSON file containing modifications
            dry_run: Preview changes without applying them
            output_file: Output path for modified spec (default: overwrite original)
            path: Project root path (default: current directory)

        Returns:
            JSON object with modification results:
            - spec_id: The specification ID
            - modifications_applied: Number of modifications applied
            - modifications_skipped: Number of modifications skipped
            - changes: Array of change summaries
            - dry_run: Whether this was a dry run
            - output_path: Path where modified spec was written
        """
        tool_name = "spec_apply_plan"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            if not modifications_file:
                return asdict(error_response(
                    "modifications_file is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a path to the modifications JSON file",
                ))

            # Build command
            cmd = ["sdd", "apply-modifications", spec_id, "--from", modifications_file, "--json"]

            if dry_run:
                cmd.append("--dry-run")

            if output_file:
                cmd.extend(["--output", output_file])

            if path:
                cmd.extend(["--path", path])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-apply-plan",
                action="apply_modifications",
                spec_id=spec_id,
                modifications_file=modifications_file,
                dry_run=dry_run,
            )

            # Execute the command with resilience
            result = _run_sdd_command(cmd, tool_name)

            # Parse the JSON output
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {}

                # Build response data
                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "modifications_applied": output_data.get("modifications_applied", output_data.get("applied", 0)),
                    "modifications_skipped": output_data.get("modifications_skipped", output_data.get("skipped", 0)),
                    "dry_run": dry_run,
                }

                # Include changes array if available
                if "changes" in output_data:
                    data["changes"] = output_data["changes"]
                elif "modifications" in output_data:
                    data["changes"] = output_data["modifications"]

                # Include output path
                if output_file:
                    data["output_path"] = output_file
                elif "output_path" in output_data:
                    data["output_path"] = output_data["output_path"]

                # Include any warnings
                warnings = []
                if output_data.get("warnings"):
                    warnings = output_data["warnings"]
                if output_data.get("modifications_skipped", 0) > 0:
                    warnings.append(f"{output_data.get('modifications_skipped', 0)} modifications were skipped")

                # Track metrics
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "success", "dry_run": str(dry_run)})

                if warnings:
                    return asdict(success_response(data, warnings=warnings))
                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})

                # Check for common errors
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        ))
                    elif "modifications" in error_msg.lower() or "file" in error_msg.lower():
                        return asdict(error_response(
                            f"Modifications file not found: {modifications_file}",
                            error_code="FILE_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the modifications file path exists",
                        ))

                if "invalid" in error_msg.lower() and "json" in error_msg.lower():
                    return asdict(error_response(
                        "Invalid modifications JSON format",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Ensure the modifications file contains valid JSON",
                    ))

                if "schema" in error_msg.lower() or "validation" in error_msg.lower():
                    return asdict(error_response(
                        f"Modifications file failed validation: {error_msg}",
                        error_code="SCHEMA_ERROR",
                        error_type="validation",
                        remediation="Check that modifications follow the expected schema",
                    ))

                return asdict(error_response(
                    f"Failed to apply modifications: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the spec exists and modifications file is valid",
                ))

        except CircuitBreakerError as e:
            return asdict(error_response(
                str(e),
                error_code="CIRCUIT_OPEN",
                error_type="unavailable",
                remediation=f"SDD CLI has failed repeatedly. Wait {e.retry_after:.0f}s before retrying.",
            ))
        except subprocess.TimeoutExpired:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "timeout"})
            return asdict(error_response(
                f"Command timed out after {CLI_TIMEOUT} seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try again or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-apply-plan")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
