"""
Documentation generation tools for foundry-mcp.

Provides MCP tools for generating human-facing documentation bundles
from SDD specifications. Uses direct Python API calls to core modules
for rendering, LLM-powered documentation, and fidelity review operations.
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import (
    get_metrics,
    mcp_tool,
)
from foundry_mcp.core.security import (
    is_prompt_injection,
    validate_size,
    MAX_STRING_LENGTH,
)
from foundry_mcp.core.spec import find_specs_directory, load_spec, find_spec_file
from foundry_mcp.core.rendering import render_spec_to_markdown, RenderOptions, RenderResult

logger = logging.getLogger(__name__)


def register_documentation_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register documentation generation tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    metrics = get_metrics()

    @canonical_tool(
        mcp,
        canonical_name="spec-doc",
    )
    @mcp_tool(tool_name="spec-doc", emit_metrics=True, audit=True)
    def spec_doc(
        spec_id: str,
        output_format: str = "markdown",
        output_path: Optional[str] = None,
        include_progress: bool = True,
        include_journal: bool = False,
        mode: str = "basic",
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Generate human-facing documentation bundle from a specification.

        Creates formatted documentation artifacts (markdown/HTML) from
        an SDD specification. Produces documentation suitable for
        stakeholders, project tracking, and archive.

        Args:
            spec_id: Specification ID to document
            output_format: Output format - 'markdown' or 'md' (HTML planned)
            output_path: Custom output path (default: specs/.human-readable/<spec_id>.md)
            include_progress: Include visual progress bars and stats
            include_journal: Include recent journal entries in output
            mode: Rendering mode - 'basic' (fast) or 'enhanced' (AI features)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with:
            - output_path: Path where documentation was written
            - format: Format of generated documentation
            - spec_id: Specification ID
            - title: Specification title
            - stats: Documentation statistics (sections, tasks, etc.)

        WHEN TO USE:
        - Generate stakeholder-friendly documentation from specs
        - Create progress reports for project tracking
        - Archive completed specifications in readable format
        - Export specs for external sharing or review
        """
        start_time = time.perf_counter()

        try:
            # Validate output_format
            if output_format not in ("markdown", "md"):
                return asdict(
                    error_response(
                        f"Invalid output_format: {output_format}. "
                        "Supported formats: 'markdown', 'md'",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use 'markdown' or 'md' as output_format.",
                    )
                )

            # Validate mode
            if mode not in ("basic", "enhanced"):
                return asdict(
                    error_response(
                        f"Invalid mode: {mode}. Must be 'basic' or 'enhanced'.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use 'basic' for fast rendering or 'enhanced' for AI features.",
                    )
                )

            # Input validation: check for prompt injection in user-controlled fields
            for field_name, field_value in [
                ("spec_id", spec_id),
                ("output_path", output_path),
                ("workspace", workspace),
            ]:
                if field_value and is_prompt_injection(field_value):
                    metrics.counter(
                        "documentation.security_blocked",
                        labels={"tool": "spec-doc", "reason": "prompt_injection"},
                    )
                    return asdict(
                        error_response(
                            f"Input validation failed for {field_name}",
                            error_code="VALIDATION_ERROR",
                            error_type="security",
                            remediation="Remove special characters or instruction-like patterns from input.",
                        )
                    )

            # Resolve workspace and find spec
            ws_path = Path(workspace) if workspace else Path.cwd()
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                metrics.counter(
                    "documentation.errors",
                    labels={"tool": "spec-doc", "error_type": "specs_dir_not_found"},
                )
                return asdict(
                    error_response(
                        "Could not find specs directory",
                        error_code="SPECS_DIR_NOT_FOUND",
                        error_type="not_found",
                        remediation="Ensure you're in a project with a specs/ directory",
                    )
                )

            # Find and load the spec
            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                metrics.counter(
                    "documentation.errors",
                    labels={"tool": "spec-doc", "error_type": "not_found"},
                )
                return asdict(
                    error_response(
                        f"Specification not found: {spec_id}",
                        error_code="SPEC_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the spec ID exists using spec-list.",
                    )
                )

            spec_data = load_spec(spec_file)

            # Build render options
            render_options = RenderOptions(
                mode=mode,
                include_progress=include_progress,
                include_journal=include_journal,
            )

            # Render the spec to markdown
            render_result: RenderResult = render_spec_to_markdown(spec_data, render_options)

            # Determine output path
            if output_path:
                final_output_path = Path(output_path)
            else:
                human_readable_dir = specs_dir / ".human-readable"
                human_readable_dir.mkdir(parents=True, exist_ok=True)
                final_output_path = human_readable_dir / f"{spec_id}.md"

            # Write the documentation
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            final_output_path.write_text(render_result.markdown, encoding="utf-8")

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.timer(
                "documentation.spec_doc_time",
                duration_ms,
                labels={"mode": mode, "format": output_format},
            )

            metrics.counter(
                "documentation.success",
                labels={"tool": "spec-doc", "mode": mode, "format": output_format},
            )
            logger.info(
                f"spec-doc completed for {spec_id}",
                extra={"spec_id": spec_id, "duration_ms": round(duration_ms, 2)},
            )

            # Build response with documentation metadata
            response_data: Dict[str, Any] = {
                "spec_id": spec_id,
                "format": output_format,
                "mode": mode,
                "output_path": str(final_output_path),
                "title": render_result.title,
                "stats": {
                    "total_tasks": render_result.total_tasks,
                    "completed_tasks": render_result.completed_tasks,
                    "total_sections": render_result.total_sections,
                },
            }

            # Add progress percentage if requested
            if include_progress and render_result.total_tasks > 0:
                response_data["progress_percentage"] = round(
                    (render_result.completed_tasks / render_result.total_tasks) * 100, 1
                )

            return asdict(
                success_response(
                    **response_data,
                    telemetry={"duration_ms": round(duration_ms, 2)},
                )
            )

        except PermissionError as e:
            metrics.counter(
                "documentation.errors",
                labels={"tool": "spec-doc", "error_type": "permission"},
            )
            return asdict(
                error_response(
                    f"Permission denied writing documentation: {str(e)}",
                    error_code="PERMISSION_DENIED",
                    error_type="validation",
                    remediation="Check file permissions for the output path.",
                )
            )
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            metrics.counter(
                "documentation.errors",
                labels={"tool": "spec-doc", "error_type": "internal"},
            )
            return asdict(
                error_response(
                    str(e),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="spec-doc-llm",
    )
    @mcp_tool(tool_name="spec-doc-llm", emit_metrics=True, audit=True)
    def spec_doc_llm(
        directory: str,
        output_dir: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        batch_size: int = 3,
        use_cache: bool = True,
        resume: bool = False,
        clear_cache: bool = False,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Generate comprehensive LLM-powered documentation for a project.

        Uses Large Language Models to analyze code and generate rich,
        context-aware documentation with explanations, examples, and
        architectural insights.

        Args:
            directory: Project directory to document
            output_dir: Output directory for documentation (default: ./docs)
            name: Project name (default: directory name)
            description: Project description for documentation context
            batch_size: Number of shards to process per batch (default: 3)
            use_cache: Enable persistent caching of parse results
            resume: Resume from previous interrupted generation
            clear_cache: Clear the cache before generating documentation
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with:
            - output_dir: Directory where documentation was generated
            - files_generated: Number of documentation files created
            - total_shards: Number of code shards processed
            - duration_seconds: Time taken for generation
            - project_name: Name of the documented project

        WHEN TO USE:
        - Generate comprehensive AI-enhanced documentation for codebases
        - Create rich developer guides with context and explanations
        - Document complex projects with architectural insights
        - Resume interrupted documentation generation
        """
        try:
            # Validate directory
            if not directory:
                return asdict(
                    error_response(
                        "Directory path is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a valid directory path to document.",
                    )
                )

            # Validate batch_size
            if batch_size < 1 or batch_size > 10:
                return asdict(
                    error_response(
                        f"Invalid batch_size: {batch_size}. Must be between 1 and 10.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use a batch_size between 1 and 10.",
                    )
                )

            # Input validation: check for prompt injection in user-controlled fields
            for field_name, field_value in [
                ("directory", directory),
                ("output_dir", output_dir),
                ("name", name),
                ("description", description),
                ("workspace", workspace),
            ]:
                if field_value and is_prompt_injection(field_value):
                    metrics.counter(
                        "documentation.security_blocked",
                        labels={"tool": "spec-doc-llm", "reason": "prompt_injection"},
                    )
                    return asdict(
                        error_response(
                            f"Input validation failed for {field_name}",
                            error_code="VALIDATION_ERROR",
                            error_type="security",
                            remediation="Remove special characters or instruction-like patterns from input.",
                        )
                    )

            # Validate description size (can be longer than other fields)
            if description:
                size_result = validate_size(
                    description, "description", max_string_length=MAX_STRING_LENGTH
                )
                if not size_result.is_valid:
                    return asdict(
                        error_response(
                            f"Description too long: {size_result.violations[0][1]}",
                            error_code="VALIDATION_ERROR",
                            error_type="validation",
                            remediation=f"Limit description to {MAX_STRING_LENGTH} characters.",
                        )
                    )

            # Validate directory exists
            dir_path = Path(directory)
            if not dir_path.exists():
                metrics.counter(
                    "documentation.errors",
                    labels={"tool": "spec-doc-llm", "error_type": "not_found"},
                )
                return asdict(
                    error_response(
                        f"Directory not found: {directory}",
                        error_code="NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the directory path exists.",
                    )
                )

            if not dir_path.is_dir():
                return asdict(
                    error_response(
                        f"Path is not a directory: {directory}",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Provide a valid directory path.",
                    )
                )

            # LLM doc generation requires external LLM integration
            # Return informative response about feature status
            metrics.counter(
                "documentation.errors",
                labels={"tool": "spec-doc-llm", "error_type": "not_implemented"},
            )
            return asdict(
                error_response(
                    "LLM documentation generation requires external LLM integration. "
                    "Use the sdd-toolkit:llm-doc-gen skill for AI-powered documentation.",
                    error_code="NOT_IMPLEMENTED",
                    error_type="unavailable",
                    data={
                        "directory": directory,
                        "alternative": "sdd-toolkit:llm-doc-gen skill",
                        "feature_status": "requires_llm_integration",
                    },
                    remediation="Use the sdd-toolkit:llm-doc-gen skill which provides "
                    "LLM-powered documentation generation with proper AI integration.",
                )
            )

        except Exception as e:
            logger.error(f"Error in spec-doc-llm: {e}")
            metrics.counter(
                "documentation.errors",
                labels={"tool": "spec-doc-llm", "error_type": "internal"},
            )
            return asdict(
                error_response(
                    str(e),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="spec-review-fidelity",
    )
    @mcp_tool(tool_name="spec-review-fidelity", emit_metrics=True, audit=True)
    def spec_review_fidelity(
        spec_id: str,
        task_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        use_ai: bool = True,
        ai_tools: Optional[List[str]] = None,
        model: Optional[str] = None,
        consensus_threshold: int = 2,
        incremental: bool = False,
        include_tests: bool = True,
        base_branch: str = "main",
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Compare implementation against specification and identify deviations.

        Performs a fidelity review to verify that code implementation matches
        the specification requirements. Uses AI consultation for comprehensive
        analysis.

        Args:
            spec_id: Specification ID to review against
            task_id: Review specific task implementation (mutually exclusive with phase_id)
            phase_id: Review entire phase implementation (mutually exclusive with task_id)
            files: Review specific file(s) only
            use_ai: Enable AI consultation for analysis (default: True)
            ai_tools: Specific AI tools to consult (default: all available)
            model: Specific model to use for AI consultation
            consensus_threshold: Minimum models that must agree (default: 2)
            incremental: Only review changed files since last run
            include_tests: Include test results in review (default: True)
            base_branch: Base branch for git diff (default: main)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with:
            - spec_id: Specification reviewed
            - scope: Review scope (task/phase/files)
            - verdict: Overall fidelity verdict (pass/partial/fail)
            - deviations: List of identified deviations
            - recommendations: Suggested fixes
            - consensus: AI model consensus information

        WHEN TO USE:
        - Verify implementation matches specification requirements
        - Check for drift between code and documented behavior
        - Review completed tasks/phases for compliance
        - Generate fidelity reports for documentation
        """
        try:
            # Validate spec_id
            if not spec_id:
                return asdict(
                    error_response(
                        "Specification ID is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a valid spec_id to review.",
                    )
                )

            # Validate mutual exclusivity of task_id and phase_id
            if task_id and phase_id:
                return asdict(
                    error_response(
                        "Cannot specify both task_id and phase_id",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Provide either task_id OR phase_id, not both.",
                    )
                )

            # Validate consensus_threshold
            if consensus_threshold < 1 or consensus_threshold > 5:
                return asdict(
                    error_response(
                        f"Invalid consensus_threshold: {consensus_threshold}. Must be between 1 and 5.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use a consensus_threshold between 1 and 5.",
                    )
                )

            # Input validation: check for prompt injection in user-controlled fields
            for field_name, field_value in [
                ("spec_id", spec_id),
                ("task_id", task_id),
                ("phase_id", phase_id),
                ("model", model),
                ("base_branch", base_branch),
                ("workspace", workspace),
            ]:
                if field_value and is_prompt_injection(field_value):
                    metrics.counter(
                        "documentation.security_blocked",
                        labels={"tool": "spec-review-fidelity", "reason": "prompt_injection"},
                    )
                    return asdict(
                        error_response(
                            f"Input validation failed for {field_name}",
                            error_code="VALIDATION_ERROR",
                            error_type="security",
                            remediation="Remove special characters or instruction-like patterns from input.",
                        )
                    )

            # Validate files array for injection
            if files:
                for idx, file_path in enumerate(files):
                    if is_prompt_injection(file_path):
                        metrics.counter(
                            "documentation.security_blocked",
                            labels={"tool": "spec-review-fidelity", "reason": "prompt_injection"},
                        )
                        return asdict(
                            error_response(
                                f"Input validation failed for files[{idx}]",
                                error_code="VALIDATION_ERROR",
                                error_type="security",
                                remediation="Remove special characters or instruction-like patterns from file paths.",
                            )
                        )

            # Resolve workspace and find spec
            ws_path = Path(workspace) if workspace else Path.cwd()
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                metrics.counter(
                    "documentation.errors",
                    labels={"tool": "spec-review-fidelity", "error_type": "specs_dir_not_found"},
                )
                return asdict(
                    error_response(
                        "Could not find specs directory",
                        error_code="SPECS_DIR_NOT_FOUND",
                        error_type="not_found",
                        remediation="Ensure you're in a project with a specs/ directory",
                    )
                )

            # Find the spec file to validate it exists
            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                metrics.counter(
                    "documentation.errors",
                    labels={"tool": "spec-review-fidelity", "error_type": "not_found"},
                )
                return asdict(
                    error_response(
                        f"Specification not found: {spec_id}",
                        error_code="SPEC_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the spec ID exists using spec-list.",
                    )
                )

            # Determine scope for informative response
            scope = "task" if task_id else ("phase" if phase_id else "spec")

            # Fidelity review requires external AI consultation
            # Return informative response about feature status
            metrics.counter(
                "documentation.errors",
                labels={"tool": "spec-review-fidelity", "error_type": "not_implemented"},
            )
            return asdict(
                error_response(
                    "Fidelity review requires external AI consultation. "
                    "Use the sdd-toolkit:sdd-fidelity-review skill for AI-powered compliance checking.",
                    error_code="NOT_IMPLEMENTED",
                    error_type="unavailable",
                    data={
                        "spec_id": spec_id,
                        "scope": scope,
                        "task_id": task_id,
                        "phase_id": phase_id,
                        "alternative": "sdd-toolkit:sdd-fidelity-review skill",
                        "feature_status": "requires_ai_integration",
                    },
                    remediation="Use the sdd-toolkit:sdd-fidelity-review skill which provides "
                    "AI-powered fidelity review with proper multi-model consultation.",
                )
            )

        except Exception as e:
            logger.error(f"Error in spec-review-fidelity: {e}")
            metrics.counter(
                "documentation.errors",
                labels={"tool": "spec-review-fidelity", "error_type": "internal"},
            )
            return asdict(
                error_response(
                    str(e),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                )
            )

    logger.debug("Registered documentation tools: spec-doc, spec-doc-llm, spec-review-fidelity")
