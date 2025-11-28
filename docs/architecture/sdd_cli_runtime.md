# ADR: SDD CLI Runtime Architecture

**Status:** Draft
**Date:** 2025-11-28
**Authors:** Claude (AI Assistant)

## Context

The foundry-mcp project provides MCP (Model Context Protocol) tools for spec-driven development. Currently, CLI functionality is provided by the separate `claude_skills` package. This ADR defines the architecture for a native CLI implementation within foundry-mcp.

### Current State

- **MCP Server:** `foundry_mcp.server` - FastMCP-based server with 80+ tools
- **Core Logic:** `foundry_mcp.core.*` - Shared business logic (spec, task, journal, etc.)
- **MCP Tools:** `foundry_mcp.tools.*` - MCP tool implementations calling core logic
- **External CLI:** `claude_skills.cli.sdd` - Separate package with CLI commands

### Goals

1. Provide native CLI as alternative to `claude_skills` for standalone usage
2. Share core logic between MCP and CLI surfaces
3. Maintain separation of concerns (transport vs business logic)
4. Enable independent testing of CLI without MCP server

## Decision

### Package Boundaries

```
foundry_mcp/
├── core/                    # Shared business logic (UNCHANGED)
│   ├── spec.py              # Spec file operations
│   ├── task.py              # Task operations
│   ├── journal.py           # Journal operations
│   ├── lifecycle.py         # Lifecycle state machine
│   ├── validation.py        # Input validation
│   ├── responses.py         # Response envelope helpers
│   └── ...
│
├── tools/                   # MCP tool implementations (UNCHANGED)
│   ├── tasks.py             # @mcp.tool decorators
│   ├── lifecycle.py         # Lifecycle MCP tools
│   └── ...
│
├── cli/                     # NEW: CLI runtime
│   ├── __init__.py          # CLI package init
│   ├── main.py              # Entry point (click app)
│   ├── commands/            # Command groups
│   │   ├── __init__.py
│   │   ├── spec.py          # sdd spec-* commands
│   │   ├── task.py          # sdd task-* commands
│   │   ├── journal.py       # sdd journal-* commands
│   │   ├── lifecycle.py     # sdd activate/complete/archive
│   │   ├── validation.py    # sdd validate/fix commands
│   │   ├── docs.py          # sdd doc-* commands
│   │   ├── test.py          # sdd test-* commands
│   │   └── review.py        # sdd review-* commands
│   ├── output.py            # JSON/text output formatting
│   ├── config.py            # CLI-specific config (work_mode, etc.)
│   └── context.py           # Session/context tracking (CLI-only)
│
├── server.py                # MCP server (UNCHANGED)
└── config.py                # Shared configuration
```

### Namespace Structure

| Namespace | Purpose | Dependencies |
|-----------|---------|--------------|
| `foundry_mcp.core` | Pure business logic | Standard library, pydantic |
| `foundry_mcp.tools` | MCP tool decorators | `foundry_mcp.core`, `mcp` |
| `foundry_mcp.cli` | CLI runtime | `foundry_mcp.core`, `click` |
| `foundry_mcp.cli.commands` | Click command groups | `foundry_mcp.cli`, `foundry_mcp.core` |

### Key Principles

1. **Core is transport-agnostic**: `foundry_mcp.core` has NO knowledge of MCP or CLI
2. **Tools and CLI are siblings**: Both depend on core, neither depends on the other
3. **Output formatting is CLI-only**: MCP uses `responses.py`, CLI uses `cli/output.py`
4. **CLI-specific features stay in CLI**: Context tracking, session markers, work mode

### Dependency Graph

```
                    ┌─────────────────┐
                    │  foundry_mcp    │
                    │    .core        │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌───────────▼───────────┐
    │  foundry_mcp      │       │   foundry_mcp         │
    │    .tools (MCP)   │       │     .cli              │
    └───────────────────┘       └───────────────────────┘
              │                             │
              │                   ┌─────────▼─────────┐
              │                   │ foundry_mcp.cli   │
              │                   │   .commands       │
              │                   └───────────────────┘
              ▼                             ▼
         MCP Server                    Click CLI
```

## Implementation Notes

### Entry Point

```python
# foundry_mcp/cli/main.py
import click
from foundry_mcp.cli.commands import spec, task, journal, lifecycle

@click.group()
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def cli(ctx, output_json):
    ctx.ensure_object(dict)
    ctx.obj['json'] = output_json

cli.add_command(spec.spec)
cli.add_command(task.task)
cli.add_command(journal.journal)
cli.add_command(lifecycle.lifecycle)

if __name__ == '__main__':
    cli()
```

### pyproject.toml Entry Point

```toml
[project.scripts]
sdd = "foundry_mcp.cli.main:cli"
```

### Command Pattern

```python
# foundry_mcp/cli/commands/task.py
import click
from foundry_mcp.core.task import get_next_task, prepare_task_context
from foundry_mcp.cli.output import output_result

@click.group()
def task():
    """Task operations."""
    pass

@task.command('next')
@click.argument('spec_id')
@click.pass_context
def next_task(ctx, spec_id):
    """Find next actionable task."""
    result = get_next_task(spec_id)
    output_result(result, json_mode=ctx.obj.get('json', False))
```

## Alternatives Considered

### Alternative 1: Flat CLI Module

```
foundry_mcp/
├── cli.py              # Single file with all commands
```

**Rejected:** Not scalable with 80+ commands. Hard to maintain.

### Alternative 2: CLI as Separate Package

```
foundry_mcp/            # MCP only
sdd_cli/                # Separate package
```

**Rejected:** Duplicates code, harder to keep in sync, more complex build.

### Alternative 3: CLI Wraps MCP Tools

```python
# CLI calls MCP tools directly
from foundry_mcp.tools.tasks import task_next
result = await task_next(spec_id=spec_id)
```

**Rejected:** MCP tools have async signatures and MCP-specific decorators. CLI should call core directly.

## Consequences

### Positive

- Single package with both MCP and CLI interfaces
- Shared core logic reduces duplication
- CLI can be tested independently of MCP server
- Clear separation of concerns

### Negative

- CLI adds `click` dependency
- CLI-specific features (context tracking) require CLI-only code
- Two surfaces to maintain for each new feature

### Neutral

- Need to ensure core functions work synchronously for CLI
- May need thin async wrappers in MCP tools over sync core functions

## Related Decisions

- **Shared Helper Strategy:** See task-1-2-2 for helper function patterns
- **Migration Priority Tiers:** See docs/cli_parity_matrix.md for implementation order
