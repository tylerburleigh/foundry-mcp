# Workflow Guide

This guide walks through a typical spec-driven development flow with both CLI
and MCP examples.

## 1) Create a spec

Create a new spec with a human-readable name. This generates a JSON spec in
`specs/pending/`.

```bash
foundry-cli specs create "Add health endpoint"
```

MCP equivalent:

```json
{
  "tool": "authoring",
  "input": {
    "action": "spec-create",
    "title": "Add health endpoint"
  }
}
```

## 2) Activate the spec

Move the spec to `specs/active/` so task execution can begin.

```bash
foundry-cli lifecycle activate <spec_id>
```

MCP equivalent:

```json
{
  "tool": "lifecycle",
  "input": {
    "action": "activate",
    "spec_id": "<spec_id>"
  }
}
```

## 3) Work tasks

Ask for the next actionable task, then mark it complete when done. Tools use
dependencies and statuses to decide what is ready.

```bash
foundry-cli tasks next <spec_id>
foundry-cli tasks complete <spec_id> <task_id> --note "Finished implementation"
```

MCP equivalents:

```json
{
  "tool": "task",
  "input": {
    "action": "next",
    "spec_id": "<spec_id>"
  }
}
```

```json
{
  "tool": "task",
  "input": {
    "action": "complete",
    "spec_id": "<spec_id>",
    "task_id": "<task_id>"
  }
}
```

## 4) Verify and review

Run validation and review steps before marking the spec complete.

```bash
foundry-cli validate check <spec_id>
foundry-cli review spec <spec_id>
```

MCP equivalents:

```json
{
  "tool": "spec",
  "input": {
    "action": "validate",
    "spec_id": "<spec_id>"
  }
}
```

```json
{
  "tool": "review",
  "input": {
    "action": "spec",
    "spec_id": "<spec_id>"
  }
}
```

## 5) Complete and archive

Once all tasks and verification are done, complete the spec. Archive when you
no longer need it in the active set.

```bash
foundry-cli lifecycle complete <spec_id>
foundry-cli lifecycle archive <spec_id>
```

MCP equivalents:

```json
{
  "tool": "lifecycle",
  "input": {
    "action": "complete",
    "spec_id": "<spec_id>"
  }
}
```

```json
{
  "tool": "lifecycle",
  "input": {
    "action": "archive",
    "spec_id": "<spec_id>"
  }
}
```

## Related guides

- [Core Concepts](02-core-concepts.md)
- [CLI Command Reference](04-cli-command-reference.md)
- [MCP Tool Reference](05-mcp-tool-reference.md)
