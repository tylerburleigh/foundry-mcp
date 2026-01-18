# First Run Example

This walkthrough creates a minimal spec, activates it, and pulls the next task.
It is designed to validate your CLI setup and specs directory.

## 1) Create a spec

```bash
foundry-cli specs create "Add health endpoint"
```

The command prints a JSON envelope with the new `spec_id`.

## 2) Activate the spec

```bash
foundry-cli lifecycle activate <spec_id>
```

## 3) Inspect phases and tasks

If you are starting from an empty template, add phases and tasks with the
authoring tools before fetching a task. If your spec already has tasks, proceed.

```bash
foundry-cli specs list-phases <spec_id>
```

## 4) Fetch the next task

```bash
foundry-cli tasks next <spec_id>
```

If no tasks are available, the response will indicate the spec is complete or
blocked. Add tasks or dependencies as needed, then re-run the command.

## 5) Mark a task complete

```bash
foundry-cli tasks complete <spec_id> <task_id> --note "Finished the task"
```

## MCP equivalent

```json
{
  "tool": "task",
  "input": {
    "action": "next",
    "spec_id": "<spec_id>"
  }
}
```
