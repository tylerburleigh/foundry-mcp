# foundry-mcp

MCP server for SDD toolkit spec management - query, navigate, and manage specification files through the Model Context Protocol.

## Overview

foundry-mcp provides an MCP (Model Context Protocol) server that enables AI assistants to interact with SDD (Spec-Driven Development) specifications. It exposes tools and resources for:

- Listing and finding specifications
- Querying task hierarchies
- Accessing spec data through MCP resources

## Installation

### Using pip

```bash
pip install foundry-mcp
```

### Using uvx (recommended for Claude Desktop)

```bash
uvx foundry-mcp
```

### From source

```bash
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp
pip install -e .
```

## Claude Desktop Configuration

Add foundry-mcp to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "foundry-mcp": {
      "command": "uvx",
      "args": ["foundry-mcp"],
      "env": {
        "FOUNDRY_MCP_SPECS_DIR": "/path/to/your/specs"
      }
    }
  }
}
```

Or if installed via pip:

```json
{
  "mcpServers": {
    "foundry-mcp": {
      "command": "foundry-mcp",
      "env": {
        "FOUNDRY_MCP_SPECS_DIR": "/path/to/your/specs"
      }
    }
  }
}
```

## Usage

### Available Tools

#### `list_specs`
List all specifications with optional status filtering.

```
list_specs(status="active")
```

#### `get_spec`
Get summary information about a specification.

```
get_spec(spec_id="my-feature-2025-01-25-001")
```

#### `get_spec_hierarchy`
Get the full task hierarchy of a specification.

```
get_spec_hierarchy(spec_id="my-feature-2025-01-25-001")
```

#### `get_task`
Get details about a specific task.

```
get_task(spec_id="my-feature-2025-01-25-001", task_id="task-1-1")
```

#### `foundry_find_specs`
Find a specification file by ID across all status folders.

```
foundry_find_specs(spec_id="my-feature-2025-01-25-001")
```

#### `foundry_list_specs`
List specifications with filtering and progress information.

```
foundry_list_specs(status="active", include_progress=True)
```

#### `foundry_query_tasks`
Query tasks within a specification by status or parent.

```
foundry_query_tasks(spec_id="my-feature-2025-01-25-001", status="pending")
```

### Available Resources

#### `specs://list`
List all specifications as an MCP resource.

#### `specs://{spec_id}`
Access a specification's full data as an MCP resource.

## Configuration

foundry-mcp can be configured via environment variables or a TOML config file.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FOUNDRY_MCP_SPECS_DIR` | Path to specs directory | Auto-detected |
| `FOUNDRY_MCP_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `FOUNDRY_MCP_API_KEYS` | Comma-separated API keys for authentication | None |
| `FOUNDRY_MCP_REQUIRE_AUTH` | Require API key authentication | false |

### TOML Configuration

Create a `foundry-mcp.toml` file:

```toml
[workspace]
specs_dir = "/path/to/specs"

[logging]
level = "INFO"
structured = true

[auth]
require_auth = false
api_keys = ["key1", "key2"]

[server]
name = "foundry-mcp"
version = "0.1.0"
```

## Development

```bash
# Clone the repository
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp

# Install in development mode
pip install -e .

# Run the server
foundry-mcp
```

## License

MIT
