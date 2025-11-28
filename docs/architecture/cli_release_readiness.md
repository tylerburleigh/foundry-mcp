# CLI Release Readiness Plan

**Status:** Draft
**Date:** 2025-11-28
**Spec:** sdd-cli-native-parity-2025-11-27-001

## Overview

This document defines the release readiness criteria and feature flag controls for the native SDD CLI implementation in foundry-mcp.

## Feature Flag Controls

### CLI Discovery Flag

Controls visibility of the native CLI in tool discovery and capabilities manifest.

```python
# foundry_mcp/core/feature_flags.py
FeatureFlag(
    name="native_cli",
    description="Enable native SDD CLI implementation",
    state=FlagState.EXPERIMENTAL,
    default_enabled=False,
    metadata={
        "category": "cli",
        "owner": "foundry-mcp",
        "documentation": "docs/architecture/sdd_cli_runtime.md"
    }
)
```

**Lifecycle Stages:**

| Stage | State | Default | Behavior |
|-------|-------|---------|----------|
| Development | `EXPERIMENTAL` | Off | CLI available only with explicit opt-in |
| Alpha | `BETA` | Off | CLI available, opt-in encouraged for testing |
| GA | `STABLE` | On | CLI enabled by default, opt-out available |
| Post-GA | - | On | Flag removed, CLI always available |

### CLI Command Group Flags

Granular flags for individual command groups during rollout:

```python
# Command group flags
CLI_FLAGS = {
    "cli_spec_commands": FeatureFlag(
        name="cli_spec_commands",
        description="Enable spec management commands (create, validate, fix)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_task_commands": FeatureFlag(
        name="cli_task_commands",
        description="Enable task management commands (next, prepare, update)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_journal_commands": FeatureFlag(
        name="cli_journal_commands",
        description="Enable journal commands (add, list)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_lifecycle_commands": FeatureFlag(
        name="cli_lifecycle_commands",
        description="Enable lifecycle commands (activate, complete, archive)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_doc_commands": FeatureFlag(
        name="cli_doc_commands",
        description="Enable documentation commands (render, doc)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_test_commands": FeatureFlag(
        name="cli_test_commands",
        description="Enable test commands (run, discover)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
    "cli_review_commands": FeatureFlag(
        name="cli_review_commands",
        description="Enable review commands (review, fidelity-review)",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        depends_on=["native_cli"]
    ),
}
```

### Capabilities Manifest Switches

The capabilities manifest advertises available CLI features:

```python
# foundry_mcp/core/capabilities.py
def get_cli_capabilities() -> dict:
    """Return CLI capabilities based on feature flags."""
    registry = get_flag_registry()

    return {
        "cli": {
            "enabled": registry.is_enabled("native_cli"),
            "version": "0.1.0",
            "command_groups": {
                "spec": registry.is_enabled("cli_spec_commands"),
                "task": registry.is_enabled("cli_task_commands"),
                "journal": registry.is_enabled("cli_journal_commands"),
                "lifecycle": registry.is_enabled("cli_lifecycle_commands"),
                "doc": registry.is_enabled("cli_doc_commands"),
                "test": registry.is_enabled("cli_test_commands"),
                "review": registry.is_enabled("cli_review_commands"),
            },
            "features": {
                "json_output": True,
                "color_output": registry.is_enabled("cli_color_output"),
                "progress_bars": registry.is_enabled("cli_progress_bars"),
            }
        }
    }
```

### Configuration Override

Users can override flags via environment or config file:

```bash
# Environment variable override
export FOUNDRY_CLI_ENABLED=true
export FOUNDRY_CLI_SPEC_COMMANDS=true

# Or via config file (.foundry/config.json)
{
    "feature_flags": {
        "native_cli": true,
        "cli_spec_commands": true,
        "cli_task_commands": true
    }
}
```

### CLI Entry Point Guard

The CLI entry point checks the feature flag:

```python
# foundry_mcp/cli/main.py
from foundry_mcp.core.feature_flags import get_flag_registry

@click.group()
@click.pass_context
def cli(ctx):
    """SDD CLI - Spec-Driven Development tools."""
    registry = get_flag_registry()

    if not registry.is_enabled("native_cli"):
        if not ctx.obj.get('force'):
            click.echo(
                "Native CLI is not yet enabled. "
                "Set FOUNDRY_CLI_ENABLED=true to opt-in.",
                err=True
            )
            ctx.exit(1)
```

## Rollout Strategy

### Phase 1: Internal Testing (Week 1-2)

- **Flag State:** `EXPERIMENTAL`
- **Default:** Off
- **Target:** Foundry-MCP developers only
- **Criteria:**
  - [ ] Core commands implemented (spec, task, journal)
  - [ ] Unit tests passing (>80% coverage)
  - [ ] No known critical bugs

### Phase 2: Alpha (Week 3-4)

- **Flag State:** `BETA`
- **Default:** Off
- **Target:** Opt-in early adopters
- **Criteria:**
  - [ ] All Tier 1 commands implemented
  - [ ] Integration tests passing
  - [ ] Documentation complete
  - [ ] Performance benchmarks met

### Phase 3: Beta (Week 5-6)

- **Flag State:** `BETA`
- **Default:** Off (moving to On)
- **Target:** Broader community testing
- **Criteria:**
  - [ ] All Tier 1 + Tier 2 commands implemented
  - [ ] No regressions from claude_skills CLI
  - [ ] User feedback incorporated
  - [ ] Migration guide published

### Phase 4: General Availability

- **Flag State:** `STABLE`
- **Default:** On
- **Target:** All users
- **Criteria:**
  - [ ] All release checklist items complete
  - [ ] No blocking issues for 2 weeks
  - [ ] Performance parity with claude_skills
  - [ ] Deprecation notice for claude_skills.cli.sdd

## Metrics and Monitoring

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Command success rate | >99% | Telemetry |
| P95 latency | <500ms | Telemetry |
| Error rate | <1% | Telemetry |
| User adoption | >50% of MCP users | Analytics |

### Rollback Triggers

Automatic rollback if:
- Error rate exceeds 5%
- P99 latency exceeds 2s
- Critical security vulnerability discovered
- Data corruption detected

### Telemetry Events

```python
# CLI telemetry events
TELEMETRY_EVENTS = [
    "cli.command.invoked",
    "cli.command.succeeded",
    "cli.command.failed",
    "cli.feature_flag.checked",
    "cli.rollback.triggered",
]
```

## Related Documents

- [CLI Runtime Architecture](sdd_cli_runtime.md) - Package structure and design
- [CLI Parity Matrix](../cli_parity_matrix.md) - Command coverage analysis
- [Feature Flags Guide](../mcp_best_practices/14-feature-flags.md) - Flag patterns
