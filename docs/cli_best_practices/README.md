# CLI Best Practices

> Guidance for building first-party SDD command-line workflows that stay consistent with MCP adapters.

## Quick Links

| Category | Documents | Focus |
|----------|-----------|-------|
| Runtime Ergonomics | [01-cli-runtime.md](./01-cli-runtime.md) | Parsers, verbosity, UX patterns |
| Command Design | [02-command-shaping.md](./02-command-shaping.md) | Namespaces, help text, alias policy |
| Shared Services | [03-shared-services.md](./03-shared-services.md) | Responses, validation, observability |
| Testing & Parity | [04-testing-parity.md](./04-testing-parity.md) | Unit coverage, fixtures, regression harness |

## How to Use This Guide

1. Start with **Runtime Ergonomics** to wire argparse, printers, and JSON output so CLI behavior matches MCP envelopes.
2. Consult **Command Design** before adding or renaming subcommands to ensure namespace alignment with canonical tool prefixes.
3. Follow **Shared Services** to reuse existing helpers (`foundry_mcp.core.*`) instead of re-inventing CLI-only utilities.
4. Apply **Testing & Parity** when validating new commands against legacy references or MCP adapters.

Each document references the relevant sections inside [docs/mcp_best_practices](../mcp_best_practices/README.md) so both surfaces share the same contracts, security posture, and rollout discipline.
