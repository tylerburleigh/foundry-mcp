"""Documentation generation commands for the SDD CLI.

This command group now provides deterministic, local documentation generation
without relying on claude_skills. It scans the repository, emits structured
artifacts (codebase.json, markdown summaries), and exposes basic cache/status
operations.
"""

import json
from datetime import datetime
from pathlib import Path
import time
from typing import List, Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)
from foundry_mcp.core.docgen import (
    DOC_ARTIFACTS,
    DocumentationGenerator,
    resolve_output_directory,
)

logger = get_cli_logger()

# Default timeout for LLM doc generation (can be long)
DOCGEN_TIMEOUT = 600  # 10 minutes


@click.group("llm-doc")
def llm_doc_group() -> None:
    """LLM-powered documentation generation commands."""
    pass


@llm_doc_group.command("generate")
@click.argument("directory")
@click.option(
    "--output-dir",
    help="Output directory for documentation (default: ./docs).",
)
@click.option(
    "--name",
    help="Project name (default: directory name).",
)
@click.option(
    "--description",
    help="Project description for documentation context.",
)
@click.option(
    "--batch-size",
    type=int,
    default=3,
    help="Number of shards to process per batch.",
)
@click.option(
    "--use-cache/--no-cache",
    default=True,
    help="Enable persistent caching of parse results.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from previous interrupted generation.",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear the cache before generating documentation.",
)
@click.pass_context
@cli_command("llm-doc-generate")
@handle_keyboard_interrupt()
@with_sync_timeout(DOCGEN_TIMEOUT, "Documentation generation timed out")
def llm_doc_generate_cmd(
    ctx: click.Context,
    directory: str,
    output_dir: Optional[str],
    name: Optional[str],
    description: Optional[str],
    batch_size: int,
    use_cache: bool,
    resume: bool,
    clear_cache: bool,
) -> None:
    """Generate project documentation artifacts."""
    start_time = time.perf_counter()
    project_root = Path(directory).expanduser().resolve()
    if not project_root.exists():
        emit_error(
            f"Project directory not found: {directory}",
            code="PROJECT_NOT_FOUND",
            error_type="validation",
            remediation="Provide a valid directory containing the project source",
            details={"directory": directory},
        )

    if batch_size <= 0:
        emit_error(
            "Batch size must be greater than zero",
            code="INVALID_BATCH_SIZE",
            error_type="validation",
            remediation="Pass --batch-size with a positive integer",
            details={"batch_size": batch_size},
        )

    destination = (
        Path(output_dir).expanduser()
        if output_dir
        else resolve_output_directory(project_root)
    )
    if not destination.is_absolute():
        destination = (project_root / destination).resolve()

    if clear_cache:
        _clear_generated_docs(destination)

    project_name = name or project_root.name
    project_description = description or f"Documentation snapshot for {project_name}"

    try:
        generator = DocumentationGenerator(project_root, destination)
    except FileNotFoundError as exc:
        emit_error(
            str(exc),
            code="PROJECT_NOT_FOUND",
            error_type="validation",
            remediation="Verify the project directory exists",
            details={"directory": directory},
        )

    result = generator.generate(
        project_name=project_name,
        description=project_description,
        use_cache=use_cache,
        resume=resume,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000
    artifacts = [artifact.to_dict() for artifact in result.artifacts]
    state_entry = next(
        (a for a in artifacts if a["name"] == "doc-generation-state.json"), None
    )

    payload = {
        "project": {
            "name": project_name,
            "root": str(project_root),
            "description": project_description,
        },
        "output_dir": str(destination),
        "statistics": result.stats.to_dict(),
        "artifacts": [
            artifact
            for artifact in artifacts
            if artifact["name"] != "doc-generation-state.json"
        ],
        "state_artifact": state_entry,
        "generation": {
            "batch_size": batch_size,
            "use_cache": use_cache,
            "resume": resume,
            "cleared_before_run": clear_cache,
        },
    }

    warnings: List[str] = []
    if batch_size != 3:
        warnings.append("Batch size hint is currently informational only.")

    emit_success(
        payload,
        warnings=warnings or None,
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


@llm_doc_group.command("status")
@click.pass_context
@cli_command("llm-doc-status")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Status lookup timed out")
def llm_doc_status_cmd(ctx: click.Context) -> None:
    """Show documentation generation status and artifacts."""
    start_time = time.perf_counter()
    project_root = _resolve_project_root(ctx)
    output_dir = resolve_output_directory(project_root)
    state = _load_generation_state(output_dir)
    artifacts = _describe_artifacts(output_dir)

    has_docs = any(artifact["exists"] for artifact in artifacts)
    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "project_root": str(project_root),
            "output_dir": str(output_dir),
            "status": "ready" if has_docs else "missing",
            "last_generated": state.get("last_updated"),
            "completed_shards": state.get("completed_shards", []),
            "artifacts": artifacts,
        },
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


@llm_doc_group.command("cache")
@click.option(
    "--action",
    type=click.Choice(["info", "clear"]),
    default="info",
    help="Show or clear generated documentation artifacts.",
)
@click.option(
    "--spec-id",
    help="Deprecated parameter retained for backwards compatibility (ignored).",
)
@click.pass_context
@cli_command("llm-doc-cache")
@handle_keyboard_interrupt()
@with_sync_timeout(30, "Cache operation timed out")
def llm_doc_cache_cmd(
    ctx: click.Context,
    action: str,
    spec_id: Optional[str],
) -> None:
    """Inspect or clear locally generated documentation artifacts."""
    start_time = time.perf_counter()
    project_root = _resolve_project_root(ctx)
    output_dir = resolve_output_directory(project_root)

    if action == "info":
        artifacts = _describe_artifacts(output_dir)
        duration_ms = (time.perf_counter() - start_time) * 1000
        emit_success(
            {
                "action": action,
                "project_root": str(project_root),
                "output_dir": str(output_dir),
                "artifacts": artifacts,
                "spec_id": spec_id,
            },
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
        return

    removed = _clear_generated_docs(output_dir)
    duration_ms = (time.perf_counter() - start_time) * 1000
    emit_success(
        {
            "action": action,
            "removed": removed,
            "output_dir": str(output_dir),
            "spec_id": spec_id,
        },
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


# Top-level alias
@click.command("generate-docs")
@click.argument("directory")
@click.option("--output-dir", help="Output directory.")
@click.option("--name", help="Project name.")
@click.option("--resume", is_flag=True, help="Resume interrupted generation.")
@click.pass_context
@cli_command("generate-docs-alias")
@handle_keyboard_interrupt()
@with_sync_timeout(DOCGEN_TIMEOUT, "Documentation generation timed out")
def generate_docs_alias_cmd(
    ctx: click.Context,
    directory: str,
    output_dir: Optional[str],
    name: Optional[str],
    resume: bool,
) -> None:
    """Generate documentation (alias for llm-doc generate)."""
    ctx.invoke(
        llm_doc_generate_cmd,
        directory=directory,
        output_dir=output_dir,
        name=name,
        description=None,
        batch_size=3,
        use_cache=True,
        resume=resume,
        clear_cache=False,
    )


def _resolve_project_root(ctx: click.Context) -> Path:
    cli_ctx = get_context(ctx)
    if cli_ctx and cli_ctx.specs_dir:
        return cli_ctx.specs_dir.parent
    return Path.cwd()


def _load_generation_state(output_dir: Path) -> dict:
    state_path = output_dir / "doc-generation-state.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _describe_artifacts(output_dir: Path) -> List[dict]:
    artifacts: List[dict] = []
    for name in DOC_ARTIFACTS:
        path = output_dir / name
        entry = {
            "name": name,
            "path": str(path),
            "exists": path.exists(),
        }
        if path.exists():
            stat = path.stat()
            entry["size_bytes"] = stat.st_size
            entry["updated_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        artifacts.append(entry)
    return artifacts


def _clear_generated_docs(output_dir: Path) -> List[str]:
    removed: List[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in DOC_ARTIFACTS:
        path = output_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed
