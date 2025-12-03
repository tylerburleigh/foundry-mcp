"""Doc-query commands for SDD CLI.

Provides commands for querying codebase documentation including:
- Finding classes and functions
- Tracing call graphs
- Analyzing code impact
- Documentation statistics
"""

import json
import time
from bisect import bisect
from typing import Optional, TYPE_CHECKING

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    SLOW_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)
from foundry_mcp.core.pagination import (
    CursorError,
    decode_cursor,
    encode_cursor,
    normalize_page_size,
)

logger = get_cli_logger()


def _validate_int_range(
    parameter: str,
    value: int,
    minimum: int,
    maximum: int,
) -> None:
    if value < minimum or value > maximum:
        emit_error(
            f"{parameter} must be between {minimum} and {maximum}",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation=f"Provide a {parameter} between {minimum} and {maximum}",
            details={"parameter": parameter, "value": value},
        )


if TYPE_CHECKING:  # pragma: no cover
    from foundry_mcp.core.docs import DocsQuery


@click.group("doc")
def doc_group() -> None:
    """Documentation query commands."""
    pass


@doc_group.command("find-class")
@click.argument("name")
@click.option(
    "--exact/--fuzzy",
    default=True,
    help="Use exact match (default) or substring match.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of results.",
)
@click.pass_context
@cli_command("doc-find-class")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Class search timed out")
def doc_find_class_cmd(
    ctx: click.Context,
    name: str,
    exact: bool,
    limit: int,
) -> None:
    """Find a class by name in codebase documentation.

    NAME is the class name to search for.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        response = query.find_class(name, exact=exact)
        if not response.success:
            emit_error(
                response.error or "Class search failed",
                code="FIND_FAILED",
                error_type="internal",
                remediation="Check that codebase documentation is valid",
                details={"query": name},
            )
            return

        raw_results = response.results
        matches = []
        for result in raw_results[: max(1, limit)]:
            match_data = dict(result.data or {})
            match = {"name": result.name}
            if result.file_path:
                match["file_path"] = result.file_path
            if result.line_number is not None:
                match["line"] = result.line_number
            match_data.pop("name", None)
            match.update(match_data)
            matches.append(match)

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                "query": name,
                "exact": exact,
                "matches": matches,
                "total_count": response.count,
                "truncated": response.count > len(matches),
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Find class failed: {e}",
            code="FIND_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={"query": name},
        )


@doc_group.command("find-function")
@click.argument("name")
@click.option(
    "--exact/--fuzzy",
    default=True,
    help="Use exact match (default) or substring match.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of results.",
)
@click.pass_context
@cli_command("doc-find-function")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Function search timed out")
def doc_find_function_cmd(
    ctx: click.Context,
    name: str,
    exact: bool,
    limit: int,
) -> None:
    """Find a function by name in codebase documentation.

    NAME is the function name to search for.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        response = query.find_function(name, exact=exact)
        if not response.success:
            emit_error(
                response.error or "Function search failed",
                code="FIND_FAILED",
                error_type="internal",
                remediation="Check that codebase documentation is valid",
                details={"query": name},
            )
            return

        raw_results = response.results
        matches = []
        for result in raw_results[: max(1, limit)]:
            match_data = dict(result.data or {})
            match = {"name": result.name}
            if result.file_path:
                match["file_path"] = result.file_path
            if result.line_number is not None:
                match["line"] = result.line_number
            if result.data.get("signature"):
                match.setdefault("signature", result.data.get("signature"))
            match_data.pop("name", None)
            match.update(match_data)
            matches.append(match)

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                "query": name,
                "exact": exact,
                "matches": matches,
                "total_count": response.count,
                "truncated": response.count > len(matches),
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Find function failed: {e}",
            code="FIND_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={"query": name},
        )


@doc_group.command("trace-calls")
@click.argument("function_name")
@click.option(
    "--direction",
    type=click.Choice(["callers", "callees", "both"]),
    default="both",
    help="Direction to trace (callers, callees, or both).",
)
@click.option(
    "--max-depth",
    type=int,
    default=3,
    help="Maximum traversal depth.",
)
@click.pass_context
@cli_command("doc-trace-calls")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Call trace timed out")
def doc_trace_calls_cmd(
    ctx: click.Context,
    function_name: str,
    direction: str,
    max_depth: int,
) -> None:
    """Trace function calls in the call graph.

    FUNCTION_NAME is the function to trace from.
    """
    _validate_int_range("max_depth", max_depth, 1, 6)
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        result = {
            "function": function_name,
            "direction": direction,
            "max_depth": max_depth,
        }

        # Trace callers (who calls this function)
        if direction in ("callers", "both"):
            callers = _trace_direction(query, function_name, "callers", max_depth)
            result["callers"] = callers

        # Trace callees (what this function calls)
        if direction in ("callees", "both"):
            callees = _trace_direction(query, function_name, "callees", max_depth)
            result["callees"] = callees

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                **result,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Trace calls failed: {e}",
            code="TRACE_FAILED",
            error_type="internal",
            remediation="Check that the function exists in codebase documentation",
            details={"function": function_name},
        )


def _trace_direction(
    query: "DocsQuery", start: str, direction: str, max_depth: int
) -> list:
    """Trace call relationships in a given direction using DocsQuery indexes."""
    visited = set()
    result = []

    def _trace(name: str, depth: int) -> None:
        if depth > max_depth or name in visited:
            return
        visited.add(name)

        if direction == "callers":
            relations = query._callers_index.get(name, [])
            for rel in relations:
                caller = query._functions_by_name.get(rel, {})
                callee = query._functions_by_name.get(name, {})
                result.append(
                    {
                        "from": rel,
                        "to": name,
                        "depth": depth,
                        "caller_file": caller.get("file"),
                        "callee_file": callee.get("file"),
                    }
                )
                _trace(rel, depth + 1)
        else:
            relations = query._callees_index.get(name, [])
            for rel in relations:
                caller = query._functions_by_name.get(name, {})
                callee = query._functions_by_name.get(rel, {})
                result.append(
                    {
                        "from": name,
                        "to": rel,
                        "depth": depth,
                        "caller_file": caller.get("file"),
                        "callee_file": callee.get("file"),
                    }
                )
                _trace(rel, depth + 1)

    _trace(start, 1)
    return result


@doc_group.command("impact")
@click.argument("target")
@click.option(
    "--type",
    "target_type",
    type=click.Choice(["class", "function", "auto"]),
    default="auto",
    help="Type of target (auto-detected by default).",
)
@click.option(
    "--max-depth",
    type=int,
    default=3,
    help="Maximum depth for impact propagation.",
)
@click.pass_context
@cli_command("doc-impact")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Impact analysis timed out")
def doc_impact_cmd(
    ctx: click.Context,
    target: str,
    target_type: str,
    max_depth: int,
) -> None:
    """Analyze impact of changing a class or function."""
    _validate_int_range("max_depth", max_depth, 1, 6)
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        detected_type = target_type
        if target_type == "auto":
            if target in query._classes_by_name:
                detected_type = "class"
            elif target in query._functions_by_name:
                detected_type = "function"
            else:
                detected_type = "unknown"

        direct_names: list[str] = []
        indirect_names: list[str] = []
        affected_files = set()
        callers_list = []

        if detected_type == "function":
            callers_list = _trace_direction(query, target, "callers", max_depth)
            for entry in callers_list:
                depth = entry.get("depth", 0)
                caller_name = entry.get("from")
                if depth == 1 and caller_name:
                    direct_names.append(caller_name)
                elif depth > 1 and caller_name:
                    indirect_names.append(caller_name)

            for name in set(direct_names + indirect_names):
                func_data = query._functions_by_name.get(name, {})
                file_path = func_data.get("file")
                if file_path:
                    affected_files.add(file_path)
        elif detected_type == "class":
            cls = query._classes_by_name.get(target, {})
            cls_file = cls.get("file")
            if cls_file:
                affected_files.add(cls_file)
                for func in query._functions_by_file.get(cls_file, []):
                    func_name = func.get("name")
                    if func_name:
                        direct_names.append(func_name)
            for other_name, other_cls in query._classes_by_name.items():
                if target in other_cls.get("bases", []):
                    direct_names.append(other_name)
                    if other_cls.get("file"):
                        affected_files.add(other_cls["file"])
        else:
            emit_error(
                f"Target not found: {target}",
                code="TARGET_NOT_FOUND",
                error_type="not_found",
                remediation="Check that the class or function name is correct",
                details={"target": target},
            )
            return

        direct_impacts = len(direct_names)
        indirect_impacts = len(indirect_names)
        impact_score = direct_impacts * 3 + indirect_impacts

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                "target": target,
                "target_type": detected_type,
                "direct_impacts": direct_impacts,
                "indirect_impacts": indirect_impacts,
                "impact_score": impact_score,
                "affected_files": sorted(affected_files),
                "callers": callers_list,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Impact analysis failed: {e}",
            code="IMPACT_FAILED",
            error_type="internal",
            remediation="Check that the target exists in codebase documentation",
            details={"target": target},
        )


@doc_group.command("stats")
@click.pass_context
@cli_command("doc-stats")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Documentation stats timed out")
def doc_stats_cmd(ctx: click.Context) -> None:
    """Get documentation statistics."""
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        class_count = len(query._classes_by_name)
        function_count = len(query._functions_by_name)
        file_count = len(
            set(query._classes_by_file.keys()) | set(query._functions_by_file.keys())
        )

        edges = set()
        for callee, callers in query._callers_index.items():
            for caller in callers:
                edges.add((caller, callee))
        total_dependencies = len(edges)

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                "class_count": class_count,
                "function_count": function_count,
                "file_count": file_count,
                "call_graph_nodes": function_count,
                "total_dependencies": total_dependencies,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Get stats failed: {e}",
            code="STATS_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={},
        )


# Top-level aliases
@click.command("find-class")
@click.argument("name")
@click.option("--exact/--fuzzy", default=True)
@click.option("--limit", type=int, default=100)
@click.pass_context
@handle_keyboard_interrupt()
def find_class_alias_cmd(
    ctx: click.Context,
    name: str,
    exact: bool,
    limit: int,
) -> None:
    """Find a class by name (alias for doc find-class)."""
    ctx.invoke(doc_find_class_cmd, name=name, exact=exact, limit=limit)


@click.command("find-function")
@click.argument("name")
@click.option("--exact/--fuzzy", default=True)
@click.option("--limit", type=int, default=100)
@click.pass_context
@handle_keyboard_interrupt()
def find_function_alias_cmd(
    ctx: click.Context,
    name: str,
    exact: bool,
    limit: int,
) -> None:
    """Find a function by name (alias for doc find-function)."""
    ctx.invoke(doc_find_function_cmd, name=name, exact=exact, limit=limit)


@doc_group.command("scope")
@click.argument("target")
@click.option(
    "--view",
    type=click.Choice(["plan", "implement", "trace", "debug"]),
    default="plan",
    help="Context view mode: plan (lightweight), implement (detailed), trace (call graph), debug (all info).",
)
@click.pass_context
@cli_command("doc-scope")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Scope query timed out")
def doc_scope_cmd(
    ctx: click.Context,
    target: str,
    view: str,
) -> None:
    """Get scoped context for a file.

    TARGET is the file path to get scope for.

    View modes:
    - plan: Lightweight context for planning (class/function names, signatures)
    - implement: Detailed context (includes docstrings, parameters, calls)
    - trace: Call graph and dependencies
    - debug: All available information
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Map view to mode
        mode = "implement" if view in ("implement", "debug") else "plan"
        result = query.get_scope(target, mode=mode)

        if not result.success:
            emit_error(
                result.error or "Scope query failed",
                code="SCOPE_FAILED",
                error_type="internal",
                remediation="Check that the file exists in documentation",
                details={"target": target},
            )
            return

        scope_data = result.results[0] if result.results else {}

        # Add trace info if requested
        if view in ("trace", "debug"):
            # Add call graph info for functions
            trace_info = []
            for func in scope_data.get("functions", []):
                func_name = func.get("name", "")
                trace_result = query.trace_calls(
                    func_name, direction="both", max_depth=2
                )
                if trace_result.success and trace_result.results:
                    trace_info.append(
                        {
                            "function": func_name,
                            "calls": [
                                {"caller": e.caller, "callee": e.callee}
                                for e in trace_result.results[:10]
                            ],
                        }
                    )
            scope_data["trace"] = trace_info

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                "target": target,
                "view": view,
                "scope": scope_data,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Scope query failed: {e}",
            code="SCOPE_FAILED",
            error_type="internal",
            remediation="Check that the target file exists in documentation",
            details={"target": target},
        )


@doc_group.command("dependencies")
@click.argument("module")
@click.option(
    "--reverse",
    is_flag=True,
    default=False,
    help="Show modules that depend on this module instead.",
)
@click.pass_context
@cli_command("doc-dependencies")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Dependencies query timed out")
def doc_dependencies_cmd(
    ctx: click.Context,
    module: str,
    reverse: bool,
) -> None:
    """Get dependencies of a module.

    MODULE is the module path (e.g., foundry_mcp.core.docs).

    Use --reverse to find modules that depend on the specified module.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        if reverse:
            result = query.get_reverse_dependencies(module)
        else:
            result = query.get_dependencies(module)

        if not result.success:
            emit_error(
                result.error or "Dependencies query failed",
                code="DEPS_FAILED",
                error_type="internal",
                remediation="Check that the module exists in documentation",
                details={"module": module},
            )
            return

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                "module": module,
                "direction": "reverse" if reverse else "forward",
                "dependencies": result.results,
                "count": len(result.results),
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Dependencies query failed: {e}",
            code="DEPS_FAILED",
            error_type="internal",
            remediation="Check that the module exists in documentation",
            details={"module": module},
        )


@doc_group.command("describe-module")
@click.argument("file_path")
@click.pass_context
@cli_command("doc-describe-module")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Describe module timed out")
def doc_describe_module_cmd(
    ctx: click.Context,
    file_path: str,
) -> None:
    """Describe a module (file) showing its classes and functions.

    FILE_PATH is the path to the file to describe.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Get classes and functions in the file
        classes_result = query.find_classes_in_file(file_path)
        functions_result = query.find_functions_in_file(file_path)

        classes = []
        for r in classes_result.results:
            cls_data = {
                "name": r.name,
                "line": r.line_number,
                "bases": r.data.get("bases", []),
                "methods": r.data.get("methods", []),
            }
            if r.data.get("docstring"):
                cls_data["docstring"] = r.data["docstring"][:200]
            classes.append(cls_data)

        functions = []
        for r in functions_result.results:
            func_data = {
                "name": r.name,
                "line": r.line_number,
                "signature": r.data.get("signature", ""),
            }
            if r.data.get("docstring"):
                func_data["docstring"] = r.data["docstring"][:200]
            functions.append(func_data)

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                "file_path": file_path,
                "classes": classes,
                "functions": functions,
                "class_count": len(classes),
                "function_count": len(functions),
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Describe module failed: {e}",
            code="DESCRIBE_FAILED",
            error_type="internal",
            remediation="Check that the file exists in documentation",
            details={"file_path": file_path},
        )


@doc_group.command("list-modules")
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of modules to return.",
)
@click.option(
    "--cursor",
    default=None,
    help="Pagination cursor from previous response.",
)
@click.pass_context
@cli_command("doc-list-modules")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "List modules timed out")
def doc_list_modules_cmd(
    ctx: click.Context,
    limit: int,
    cursor: Optional[str],
) -> None:
    """List all modules (files) in the codebase documentation."""
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    limit = normalize_page_size(limit)

    start_after = None
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
        except CursorError as exc:
            emit_error(
                "Invalid pagination cursor",
                code="INVALID_CURSOR",
                error_type="validation",
                remediation="Pass the cursor value from the previous response or omit --cursor.",
                details={"reason": exc.reason or "decode_failed"},
            )
            return
        if cursor_data.get("mode") != "doc_modules":
            emit_error(
                "Cursor not valid for doc list-modules",
                code="INVALID_CURSOR",
                error_type="validation",
                remediation="Restart without --cursor to obtain a new cursor.",
                details={
                    "mode": cursor_data.get("mode"),
                    "expected_mode": "doc_modules",
                },
            )
            return
        start_after = cursor_data.get("last_file")

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        files = set()
        for cls in query._classes_by_name.values():
            if cls.get("file"):
                files.add(cls["file"])
        for func in query._functions_by_name.values():
            if func.get("file"):
                files.add(func["file"])

        sorted_files = sorted(files)
        start_index = 0
        if start_after:
            start_index = bisect(sorted_files, start_after)

        window = sorted_files[start_index : start_index + limit + 1]
        page_files = window[:limit]
        has_more = len(window) > limit

        modules = []
        for file_path in page_files:
            class_count = len(query._classes_by_file.get(file_path, []))
            function_count = len(query._functions_by_file.get(file_path, []))
            modules.append(
                {
                    "file_path": file_path,
                    "class_count": class_count,
                    "function_count": function_count,
                }
            )

        next_cursor = None
        if has_more and modules:
            next_cursor = encode_cursor(
                {
                    "mode": "doc_modules",
                    "last_file": modules[-1]["file_path"],
                }
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        pagination = {
            "cursor": next_cursor,
            "has_more": has_more,
            "page_size": limit,
        }

        emit_success(
            {
                "modules": modules,
                "count": len(modules),
                "total_files": len(files),
                "offset": start_index,
                "truncated": has_more,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            },
            pagination=pagination,
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"List modules failed: {e}",
            code="LIST_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={},
        )


@doc_group.command("list-functions")
@click.option(
    "--file",
    "file_path",
    default=None,
    help="Filter to functions in a specific file.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of functions to return.",
)
@click.option(
    "--cursor",
    default=None,
    help="Pagination cursor from previous response.",
)
@click.pass_context
@cli_command("doc-list-functions")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "List functions timed out")
def doc_list_functions_cmd(
    ctx: click.Context,
    file_path: Optional[str],
    limit: int,
    cursor: Optional[str],
) -> None:
    """List functions in the codebase documentation.

    Use --file to filter to a specific file.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    limit = normalize_page_size(limit)

    scope = file_path or "__all__"
    cursor_name: Optional[str] = None
    cursor_file: Optional[str] = None
    cursor_line: Optional[int] = None

    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
        except CursorError as exc:
            emit_error(
                "Invalid pagination cursor",
                code="INVALID_CURSOR",
                error_type="validation",
                remediation="Pass the cursor value from the previous response or omit --cursor.",
                details={"reason": exc.reason or "decode_failed"},
            )
            return
        if (
            cursor_data.get("mode") != "doc_functions"
            or cursor_data.get("scope") != scope
        ):
            emit_error(
                "Cursor not valid for doc list-functions",
                code="INVALID_CURSOR",
                error_type="validation",
                remediation="Restart without --cursor to obtain a new cursor.",
                details={
                    "mode": cursor_data.get("mode"),
                    "scope": cursor_data.get("scope"),
                    "expected_scope": scope,
                },
            )
            return
        cursor_name = cursor_data.get("last_name")
        cursor_file = cursor_data.get("last_file")
        cursor_line_value = cursor_data.get("last_line")
        if cursor_line_value is not None:
            try:
                cursor_line = int(cursor_line_value)
            except ValueError:
                cursor_line = None

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        def build_entries(results):
            rows = []
            for r in results:
                rows.append(
                    {
                        "name": r.name,
                        "file_path": r.file_path,
                        "line": r.line_number,
                        "signature": r.data.get("signature", ""),
                    }
                )
            return rows

        if file_path:
            result = query.find_functions_in_file(file_path)
            all_entries = build_entries(result.results)
        else:
            all_entries = []
            for func_name, func in sorted(query._functions_by_name.items()):
                all_entries.append(
                    {
                        "name": func_name,
                        "file_path": func.get("file"),
                        "line": func.get("line"),
                        "signature": func.get("signature", ""),
                    }
                )

        total_count = len(all_entries)
        start_index = 0
        if cursor_name:
            match_index = None
            for idx, entry in enumerate(all_entries):
                if (
                    entry.get("name") == cursor_name
                    and entry.get("file_path") == cursor_file
                    and entry.get("line") == cursor_line
                ):
                    match_index = idx
                    break
            if match_index is None:
                emit_error(
                    "Cursor no longer valid for doc list-functions",
                    code="INVALID_CURSOR",
                    error_type="validation",
                    remediation="Restart without --cursor to obtain a new cursor.",
                    details={"reason": "target_not_found"},
                )
                return
            start_index = match_index + 1

        window = all_entries[start_index : start_index + limit + 1]
        page_entries = window[:limit]
        has_more = len(window) > limit

        next_cursor = None
        if has_more and page_entries:
            last_entry = page_entries[-1]
            next_cursor = encode_cursor(
                {
                    "mode": "doc_functions",
                    "scope": scope,
                    "last_name": last_entry.get("name"),
                    "last_file": last_entry.get("file_path"),
                    "last_line": last_entry.get("line"),
                }
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        pagination = {
            "cursor": next_cursor,
            "has_more": has_more,
            "page_size": limit,
        }

        emit_success(
            {
                "functions": page_entries,
                "count": len(page_entries),
                "total_count": total_count,
                "file_filter": file_path,
                "offset": start_index,
                "truncated": has_more,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            },
            pagination=pagination,
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"List functions failed: {e}",
            code="LIST_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={},
        )


@doc_group.command("search")
@click.argument("query")
@click.option(
    "--type",
    "entity_type",
    type=click.Choice(["class", "function", "all"]),
    default="all",
    help="Type of entity to search for.",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="Maximum number of results.",
)
@click.option(
    "--cursor",
    default=None,
    help="Pagination cursor from previous response.",
)
@click.pass_context
@cli_command("doc-search")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Search timed out")
def doc_search_cmd(
    ctx: click.Context,
    query: str,
    entity_type: str,
    limit: int,
    cursor: Optional[str],
) -> None:
    """Search documentation by keyword.

    QUERY is the search term (case-insensitive substring match).

    Searches class names, function names, docstrings, and method names.
    Results are sorted by relevance score.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    limit = normalize_page_size(limit)

    offset = 0
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
        except CursorError as exc:
            emit_error(
                "Invalid pagination cursor",
                code="INVALID_CURSOR",
                error_type="validation",
                remediation="Pass the cursor value from the previous response or omit --cursor.",
                details={"reason": exc.reason or "decode_failed"},
            )
            return
        if (
            cursor_data.get("mode") != "doc_search"
            or cursor_data.get("query") != query
            or cursor_data.get("entity_type") != entity_type
        ):
            emit_error(
                "Cursor not valid for doc search",
                code="INVALID_CURSOR",
                error_type="validation",
                remediation="Restart without --cursor to obtain a new cursor.",
                details={
                    "mode": cursor_data.get("mode"),
                    "query": cursor_data.get("query"),
                    "entity_type": cursor_data.get("entity_type"),
                },
            )
            return
        offset = int(cursor_data.get("offset", 0))
        if offset < 0:
            offset = 0

    try:
        from foundry_mcp.core.docs import DocsQuery

        query_obj = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query_obj.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        entity_types = None
        if entity_type == "class":
            entity_types = ["class"]
        elif entity_type == "function":
            entity_types = ["function"]

        result = query_obj.search(
            query, entity_types=entity_types, max_results=offset + limit + 1
        )

        if not result.success:
            emit_error(
                result.error or "Search failed",
                code="SEARCH_FAILED",
                error_type="internal",
                remediation="Check that the query is valid",
                details={"query": query},
            )
            return

        raw_matches = result.results
        page_window = raw_matches[offset : offset + limit + 1]
        page_results = page_window[:limit]
        has_more = len(raw_matches) > offset + limit

        matches = []
        for r in page_results:
            match = {
                "name": r.name,
                "type": r.entity_type,
                "file_path": r.file_path,
                "line": r.line_number,
                "relevance": round(r.relevance_score, 2),
            }
            if r.data.get("signature"):
                match["signature"] = r.data["signature"]
            matches.append(match)

        next_cursor = None
        if has_more and matches:
            next_cursor = encode_cursor(
                {
                    "mode": "doc_search",
                    "query": query,
                    "entity_type": entity_type,
                    "offset": offset + len(matches),
                }
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        pagination = {
            "cursor": next_cursor,
            "has_more": has_more,
            "page_size": limit,
        }

        emit_success(
            {
                "query": query,
                "type_filter": entity_type,
                "matches": matches,
                "count": len(matches),
                "offset": offset,
                "truncated": has_more,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            },
            pagination=pagination,
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Search failed: {e}",
            code="SEARCH_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={"query": query},
        )


@doc_group.command("context")
@click.argument("target")
@click.option(
    "--depth",
    type=int,
    default=2,
    help="Depth of context to include (1=minimal, 2=standard, 3=comprehensive).",
)
@click.pass_context
@cli_command("doc-context")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Context query timed out")
def doc_context_cmd(
    ctx: click.Context,
    target: str,
    depth: int,
) -> None:
    """Get comprehensive context for a class or function.

    TARGET is the name of a class or function.

    Provides context including the entity definition, callers/callees,
    file context, and related entities.
    """
    _validate_int_range("depth", depth, 1, 3)
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Try to find as class first, then function
        entity_type = None
        entity_data = None

        cls_result = query.find_class(target, exact=True)
        if cls_result.success and cls_result.results:
            entity_type = "class"
            entity_data = cls_result.results[0]
        else:
            func_result = query.find_function(target, exact=True)
            if func_result.success and func_result.results:
                entity_type = "function"
                entity_data = func_result.results[0]

        if not entity_data:
            emit_error(
                f"Target not found: {target}",
                code="TARGET_NOT_FOUND",
                error_type="not_found",
                remediation="Check that the class or function name is correct",
                details={"target": target},
            )
            return

        context_data = {
            "name": entity_data.name,
            "type": entity_type,
            "file_path": entity_data.file_path,
            "line": entity_data.line_number,
        }

        # Add entity-specific data
        if entity_type == "class":
            context_data["bases"] = entity_data.data.get("bases", [])
            context_data["methods"] = entity_data.data.get("methods", [])
            if entity_data.data.get("docstring"):
                context_data["docstring"] = entity_data.data["docstring"][:500]
        else:
            context_data["signature"] = entity_data.data.get("signature", "")
            if entity_data.data.get("docstring"):
                context_data["docstring"] = entity_data.data["docstring"][:500]
            context_data["parameters"] = entity_data.data.get("parameters", [])

        # Add call graph info for functions (depth >= 2)
        if entity_type == "function" and depth >= 2:
            callers_result = query.get_callers(target)
            if callers_result.success:
                context_data["callers"] = [
                    {"name": r.name, "file": r.file_path}
                    for r in callers_result.results[:10]
                ]

            callees_result = query.get_callees(target)
            if callees_result.success:
                context_data["callees"] = [
                    {"name": r.name, "file": r.file_path}
                    for r in callees_result.results[:10]
                ]

        # Add file context (depth >= 2)
        if depth >= 2 and entity_data.file_path:
            file_classes = query.find_classes_in_file(entity_data.file_path)
            file_funcs = query.find_functions_in_file(entity_data.file_path)

            context_data["file_context"] = {
                "classes": [r.name for r in file_classes.results if r.name != target],
                "functions": [r.name for r in file_funcs.results if r.name != target],
            }

        # Add impact analysis (depth >= 3)
        if depth >= 3:
            impact_target_type = entity_type if entity_type is not None else "auto"
            impact_result = query.impact_analysis(
                target, target_type=impact_target_type
            )
            if impact_result.success and impact_result.results:
                impact = impact_result.results[0]
                context_data["impact"] = {
                    "direct_impacts": impact.direct_impacts[:10],
                    "indirect_impacts": impact.indirect_impacts[:10],
                    "impact_score": impact.impact_score,
                    "affected_files": impact.affected_files[:10],
                }

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success(
            {
                "target": target,
                "depth": depth,
                "context": context_data,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            }
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Context query failed: {e}",
            code="CONTEXT_FAILED",
            error_type="internal",
            remediation="Check that the target exists in documentation",
            details={"target": target},
        )


@doc_group.command("refactor-candidates")
@click.option(
    "--threshold",
    type=int,
    default=3,
    help="Minimum number of callers to flag as candidate.",
)
@click.option(
    "--complexity",
    type=int,
    default=0,
    help="Minimum complexity score to include.",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="Maximum number of candidates to return.",
)
@click.option(
    "--cursor",
    default=None,
    help="Pagination cursor from previous response.",
)
@click.pass_context
@cli_command("doc-refactor-candidates")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Refactor candidates query timed out")
def doc_refactor_candidates_cmd(
    ctx: click.Context,
    threshold: int,
    complexity: int,
    limit: int,
    cursor: Optional[str],
) -> None:
    """Find candidates for refactoring.

    Identifies functions and classes that may benefit from refactoring based on:
    - High caller count (potential utility extraction)
    - High callee count (complex functions)
    - High complexity score (if available)
    - Large classes (many methods)
    - Deep inheritance hierarchies

    Results are sorted by priority score.
    """
    _validate_int_range("threshold", threshold, 0, 1000)
    _validate_int_range("complexity", complexity, 0, 500)
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    limit = normalize_page_size(limit)

    offset = 0
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
        except CursorError as exc:
            emit_error(
                "Invalid pagination cursor",
                code="INVALID_CURSOR",
                error_type="validation",
                remediation="Pass the cursor value from the previous response or omit --cursor.",
                details={"reason": exc.reason or "decode_failed"},
            )
            return
        if (
            cursor_data.get("mode") != "doc_refactor"
            or int(cursor_data.get("threshold", threshold)) != threshold
            or int(cursor_data.get("complexity", complexity)) != complexity
        ):
            emit_error(
                "Cursor not valid for doc refactor-candidates",
                code="INVALID_CURSOR",
                error_type="validation",
                remediation="Restart without --cursor to obtain a new cursor.",
                details={
                    "mode": cursor_data.get("mode"),
                    "threshold": cursor_data.get("threshold"),
                    "complexity": cursor_data.get("complexity"),
                },
            )
            return
        offset = int(cursor_data.get("offset", 0))
        if offset < 0:
            offset = 0

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(
            workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None
        )
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        result = query.get_refactor_candidates(
            min_callers=threshold, min_complexity=complexity
        )

        if not result.success:
            emit_error(
                result.error or "Refactor candidates query failed",
                code="REFACTOR_FAILED",
                error_type="internal",
                remediation="Check that codebase documentation is valid",
                details={},
            )
            return

        total_found = len(result.results)
        if offset > total_found:
            emit_error(
                "Cursor exceeds available results",
                code="INVALID_CURSOR",
                error_type="validation",
                remediation="Restart without --cursor to obtain a new cursor.",
                details={"offset": offset, "total": total_found},
            )
            return

        window = result.results[offset : offset + limit + 1]
        page_candidates = window[:limit]
        has_more = len(window) > limit

        next_cursor = None
        if has_more and page_candidates:
            next_cursor = encode_cursor(
                {
                    "mode": "doc_refactor",
                    "threshold": threshold,
                    "complexity": complexity,
                    "offset": offset + len(page_candidates),
                }
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        pagination = {
            "cursor": next_cursor,
            "has_more": has_more,
            "page_size": limit,
        }

        emit_success(
            {
                "threshold": threshold,
                "complexity_min": complexity,
                "candidates": page_candidates,
                "count": len(page_candidates),
                "total_found": total_found,
                "offset": offset,
                "truncated": has_more,
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            },
            pagination=pagination,
        )

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Refactor candidates query failed: {e}",
            code="REFACTOR_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={},
        )
