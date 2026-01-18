# Quick Start

Get foundry-mcp running and verify the CLI and MCP server in a few minutes.

## What you will do

- Install the package
- Run the CLI once
- Start the MCP server and call a health check

## Prerequisites

- Python 3.10+
- An MCP-compatible client if you plan to use the MCP server

## Step 1: Install foundry-mcp

Pick one of the install paths:

```bash
pip install foundry-mcp
```

```bash
uvx foundry-mcp
```

## Step 2: Verify the CLI

```bash
foundry-cli --help
```

You can also run the module directly:

```bash
python -m foundry_mcp.cli --help
```

## Step 3: Start the MCP server

```bash
foundry-mcp
```

In your MCP client, call the health tool to verify connectivity:

```json
{
  "tool": "health",
  "input": {
    "action": "liveness"
  }
}
```

## Step 4: Point foundry-mcp at your specs

foundry-mcp looks for a `specs/` directory in the workspace. You can also
explicitly configure it:

```toml
# foundry-mcp.toml
[workspace]
specs_dir = "./specs"
```

Or set an environment variable:

```bash
export FOUNDRY_MCP_SPECS_DIR="$PWD/specs"
```

## Next steps

- Learn the core model in [Core Concepts](02-core-concepts.md)
- Walk through a full flow in [Workflow Guide](03-workflow-guide.md)
- Configure providers in [Configuration](06-configuration.md)
