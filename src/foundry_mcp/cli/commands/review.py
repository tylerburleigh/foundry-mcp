"""Review commands for SDD CLI.

Provides commands for structural spec review metadata and exposes
availability of LLM-powered review workflows.
"""

from dataclasses import asdict
import time
from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    SLOW_TIMEOUT,
    MEDIUM_TIMEOUT,
    with_sync_timeout,
    handle_keyboard_interrupt,
)
from foundry_mcp.core.review import quick_review, review_type_requires_llm

logger = get_cli_logger()

# Review types supported
REVIEW_TYPES = ["quick", "full", "security", "feasibility"]

REVIEW_TOOL_DEFINITIONS = [
    {
        "name": "quick-review",
        "description": "Structural validation with schema & progress checks (native).",
        "capabilities": ["structure", "progress", "quality"],
        "requires_llm": False,
    },
    {
        "name": "full-review",
        "description": "LLM-powered deep review via sdd-toolkit.",
        "capabilities": ["structure", "quality", "suggestions"],
        "requires_llm": True,
        "alternative": "sdd-toolkit:sdd-plan-review",
    },
    {
        "name": "security-review",
        "description": "Security-focused LLM analysis.",
        "capabilities": ["security", "trust_boundaries"],
        "requires_llm": True,
        "alternative": "sdd-toolkit:sdd-plan-review",
    },
    {
        "name": "feasibility-review",
        "description": "Implementation feasibility assessment (LLM).",
        "capabilities": ["complexity", "risk", "dependencies"],
        "requires_llm": True,
        "alternative": "sdd-toolkit:sdd-plan-review",
    },
]

# Fidelity review timeout (longer for AI consultation)
FIDELITY_TIMEOUT = 600


@click.group("review")
def review_group() -> None:
    """Spec review and fidelity checking commands."""
    pass


@review_group.command("spec")
@click.argument("spec_id")
@click.option(
    "--type",
    "review_type",
    type=click.Choice(REVIEW_TYPES),
    default="quick",
    help="Type of review to perform.",
)
@click.option(
    "--tools",
    help="Comma-separated list of review tools to use (LLM types only).",
)
@click.option(
    "--model",
    help="LLM model to use for review (LLM types only).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be reviewed without executing.",
)
@click.pass_context
@cli_command("review-spec")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Review timed out")
def review_spec_cmd(
    ctx: click.Context,
    spec_id: str,
    review_type: str,
    tools: Optional[str],
    model: Optional[str],
    dry_run: bool,
) -> None:
    """Run a structural review or report availability of LLM reviews."""
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    llm_status = _get_llm_status()

    if review_type_requires_llm(review_type):
        emit_error(
            f"Review type '{review_type}' is handled by the sdd-toolkit review workflow",
            code="NOT_IMPLEMENTED",
            error_type="unavailable",
            remediation="Use the sdd-toolkit:sdd-plan-review skill for LLM-powered reviews",
            details={
                "spec_id": spec_id,
                "review_type": review_type,
                "alternative": "sdd-toolkit:sdd-plan-review",
                "llm_status": llm_status,
            },
        )

    if dry_run:
        emit_success(
            {
                "spec_id": spec_id,
                "review_type": review_type,
                "dry_run": True,
                "llm_status": llm_status,
                "message": "Dry run - quick review skipped",
            }
        )
        return

    quick_result = quick_review(spec_id=spec_id, specs_dir=specs_dir)
    duration_ms = (time.perf_counter() - start_time) * 1000

    payload = asdict(quick_result)
    payload["llm_status"] = llm_status

    emit_success(
        payload,
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


@review_group.command("tools")
@click.pass_context
@cli_command("review-tools")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Review tools lookup timed out")
def review_tools_cmd(ctx: click.Context) -> None:
    """List native and external review toolchains."""
    start_time = time.perf_counter()

    llm_status = _get_llm_status()

    tools_info = []
    for definition in REVIEW_TOOL_DEFINITIONS:
        requires_llm = definition.get("requires_llm", False)
        available = not requires_llm  # LLM reviews are handled by external workflows
        tool_info = {
            "name": definition["name"],
            "description": definition["description"],
            "capabilities": definition.get("capabilities", []),
            "requires_llm": requires_llm,
            "available": available,
            "status": "native" if available else "external",
        }
        if not available:
            tool_info["alternative"] = definition.get("alternative")
            tool_info["message"] = "Use the sdd-toolkit workflow for this review type"
        tools_info.append(tool_info)

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "tools": tools_info,
            "llm_status": llm_status,
            "review_types": REVIEW_TYPES,
        },
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


@review_group.command("plan-tools")
@click.pass_context
@cli_command("review-plan-tools")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Plan tools lookup timed out")
def review_plan_tools_cmd(ctx: click.Context) -> None:
    """List available plan review toolchains."""
    start_time = time.perf_counter()

    llm_status = _get_llm_status()

    # Define plan review toolchains
    plan_tools = [
        {
            "name": "quick-review",
            "description": "Fast structural review for basic validation",
            "capabilities": ["structure", "syntax", "basic_quality"],
            "llm_required": False,
            "estimated_time": "< 10 seconds",
        },
        {
            "name": "full-review",
            "description": "Comprehensive review with LLM analysis",
            "capabilities": ["structure", "quality", "feasibility", "suggestions"],
            "llm_required": True,
            "estimated_time": "30-60 seconds",
        },
        {
            "name": "security-review",
            "description": "Security-focused analysis of plan",
            "capabilities": ["security", "trust_boundaries", "data_flow"],
            "llm_required": True,
            "estimated_time": "30-60 seconds",
        },
        {
            "name": "feasibility-review",
            "description": "Implementation feasibility assessment",
            "capabilities": ["complexity", "dependencies", "risk"],
            "llm_required": True,
            "estimated_time": "30-60 seconds",
        },
    ]

    # Add availability status (only quick review is native today)
    available_tools = []
    for tool in plan_tools:
        tool_info = tool.copy()
        if tool["llm_required"]:
            tool_info["status"] = "external"
            tool_info["available"] = False
            tool_info["reason"] = "Use the sdd-toolkit:sdd-plan-review workflow"
            tool_info["alternative"] = "sdd-toolkit:sdd-plan-review"
        else:
            tool_info["status"] = "native"
            tool_info["available"] = True
        available_tools.append(tool_info)

    recommendations = [
        "Use 'quick-review' for structural validation inside foundry-mcp",
        "Invoke sdd-toolkit:sdd-plan-review for AI-assisted plan analysis",
        "Configure LLM credentials when ready to adopt the toolkit workflow",
    ]

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "plan_tools": available_tools,
            "llm_status": llm_status,
            "recommendations": recommendations,
        },
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


@review_group.command("fidelity")
@click.argument("spec_id")
@click.option(
    "--task",
    "task_id",
    help="Review specific task implementation.",
)
@click.option(
    "--phase",
    "phase_id",
    help="Review entire phase implementation.",
)
@click.option(
    "--files",
    multiple=True,
    help="Review specific file(s) only.",
)
@click.option(
    "--incremental",
    is_flag=True,
    help="Only review changed files since last run.",
)
@click.option(
    "--base-branch",
    default="main",
    help="Base branch for git diff.",
)
@click.pass_context
@cli_command("review-fidelity")
@handle_keyboard_interrupt()
@with_sync_timeout(FIDELITY_TIMEOUT, "Fidelity review timed out")
def review_fidelity_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: Optional[str],
    phase_id: Optional[str],
    files: tuple,
    incremental: bool,
    base_branch: str,
) -> None:
    """Compare implementation against specification.

    SPEC_ID is the specification identifier.

    Performs a fidelity review to verify that code implementation
    matches the specification requirements.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Validate mutually exclusive options
    if task_id and phase_id:
        emit_error(
            "Cannot specify both --task and --phase",
            code="INVALID_OPTIONS",
            error_type="validation",
            remediation="Use either --task or --phase, not both",
            details={"hint": "Use either --task or --phase, not both"},
        )

    llm_status = _get_llm_status()

    scope = "spec"
    if task_id:
        scope = f"task:{task_id}"
    elif phase_id:
        scope = f"phase:{phase_id}"
    elif files:
        scope = f"files:{len(files)}"

    emit_error(
        "Fidelity review requires the sdd-toolkit fidelity workflow",
        code="NOT_IMPLEMENTED",
        error_type="unavailable",
        remediation="Use the sdd-toolkit:sdd-fidelity-review skill for implementation comparisons",
        details={
            "spec_id": spec_id,
            "scope": scope,
            "alternative": "sdd-toolkit:sdd-fidelity-review",
            "llm_status": llm_status,
            "incremental": incremental,
            "base_branch": base_branch,
        },
    )


def _get_llm_status() -> dict:
    """Get LLM configuration status."""
    try:
        from foundry_mcp.core.llm_config import get_llm_config

        config = get_llm_config()
        return {
            "configured": config.get_api_key() is not None,
            "provider": config.provider.value,
            "model": config.get_model(),
        }
    except ImportError:
        return {"configured": False, "error": "LLM config not available"}
    except Exception as e:
        return {"configured": False, "error": str(e)}


# Top-level aliases
@click.command("review-spec")
@click.argument("spec_id")
@click.option(
    "--type",
    "review_type",
    type=click.Choice(REVIEW_TYPES),
    default="quick",
    help="Type of review to perform.",
)
@click.pass_context
@cli_command("review-spec-alias")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Review timed out")
def review_spec_alias_cmd(
    ctx: click.Context,
    spec_id: str,
    review_type: str,
) -> None:
    """Run an LLM-powered review on a specification (alias)."""
    # Delegate to main command
    ctx.invoke(review_spec_cmd, spec_id=spec_id, review_type=review_type)
