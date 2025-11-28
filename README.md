# foundry-mcp

MCP server for SDD toolkit spec management - query, navigate, and manage specification files through the Model Context Protocol.

## Overview

foundry-mcp provides a comprehensive MCP (Model Context Protocol) server that enables AI assistants to interact with SDD (Spec-Driven Development) specifications. It provides 40+ tools, resources, and prompts for:

- **Spec Management**: List, find, and navigate specifications
- **Task Operations**: Track progress, manage dependencies, prepare task context
- **Validation & Fixes**: Validate specs and auto-fix common issues
- **Journaling**: Track decisions, blockers, and status changes
- **Lifecycle Management**: Move specs through pending → active → completed → archived
- **Code Documentation**: Query codebase docs, trace call graphs, analyze impact
- **Testing**: Run pytest with presets, discover tests, track results
- **Resources & Prompts**: Access specs via URIs, use workflow prompts

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

## Tools Reference

Tool names follow the [Tool Naming Conventions](docs/codebase_standards/naming-conventions.md); consult that guide before adding or renaming MCP operations.

### Spec Management

| Tool | Description |
|------|-------------|
| `list_specs` | List all specs with optional status filtering |
| `get_spec` | Get summary info about a specification |
| `get_spec_hierarchy` | Get full task hierarchy |
| `get_task` | Get details about a specific task |

### Query Tools

| Tool | Description |
|------|-------------|
| `task_query` | Query tasks by status, type, or parent |

### Task Operations

| Tool | Description |
|------|-------------|
| `task_prepare` | Prepare task with full context (dependencies, sibling info, phase) |
| `task_next` | Find next actionable task based on status and dependencies |
| `task_info` | Get detailed task information |
| `task_check_deps` | Check dependency status for a task |
| `task_update_status` | Update task status (pending, in_progress, completed, blocked) |
| `task_complete` | Mark task complete with journal entry |
| `task_start` | Start working on a task |
| `task_progress` | Get spec/phase progress summary |

### Validation Tools

| Tool | Description |
|------|-------------|
| `spec_validate` | Validate spec and return structured diagnostics |
| `spec_fix` | Apply auto-fixes with dry-run support |
| `spec_stats` | Get comprehensive spec statistics |
| `spec_validate_fix` | Combined validation and fix in one operation |

### Journal Tools

| Tool | Description |
|------|-------------|
| `journal_add` | Add journal entries with optional task association |
| `journal_list` | Retrieve journal entries with filtering |
| `task_block` | Mark tasks as blocked with metadata |
| `task_unblock` | Unblock tasks and track resolution |
| `task_list_blocked` | List all blocked tasks |
| `journal_list_unjournaled` | Find tasks needing journal entries |

### Rendering Tools

| Tool | Description |
|------|-------------|
| `spec_render` | Render spec to markdown with mode options |
| `spec_render_progress` | ASCII progress bars for spec and phases |
| `task_list` | Flat task list with filtering |

### Lifecycle Tools

| Tool | Description |
|------|-------------|
| `spec_lifecycle_move` | Move spec between folders |
| `spec_lifecycle_activate` | Activate pending spec (pending → active) |
| `spec_lifecycle_complete` | Mark spec completed (active → completed) |
| `spec_lifecycle_archive` | Archive spec (completed → archived) |
| `spec_lifecycle_state` | Get lifecycle state and transition eligibility |
| `spec_list_by_folder` | List specs organized by folder |

### Documentation Tools

| Tool | Description |
|------|-------------|
| `code_find_class` | Find class in codebase documentation |
| `code_find_function` | Find function in documentation |
| `code_trace_calls` | Trace call graph (callers/callees) with depth |
| `code_impact_analysis` | Analyze change impact across codebase |
| `code_get_callers` | Get caller functions |
| `code_get_callees` | Get callee functions |
| `doc_stats` | Get documentation statistics |

### Testing Tools

| Tool | Description |
|------|-------------|
| `test_run` | Full-featured test execution with presets |
| `test_discover` | Test discovery without execution |
| `test_presets` | List available test presets |
| `test_run_quick` | Quick test preset (60s timeout, fail-fast) |
| `test_run_unit` | Unit test preset |

## Resources

Access spec data through MCP resources:

| Resource URI | Description |
|--------------|-------------|
| `foundry://specs/` | List all specifications |
| `foundry://specs/{status}/` | List specs by status (active, pending, completed, archived) |
| `foundry://specs/{status}/{spec_id}` | Get specific spec with full hierarchy |
| `foundry://specs/{spec_id}/journal` | Get journal entries for a spec |
| `foundry://templates/` | List available spec templates |
| `foundry://templates/{template_id}` | Get specific template |

### Built-in Templates

- **basic**: Minimal spec with single phase
- **feature**: Standard feature with design, implementation, verification phases
- **bugfix**: Bug investigation, fix, and verification

## Prompts

Workflow prompts for common operations:

| Prompt | Description |
|--------|-------------|
| `start_feature` | Guide new feature setup with template selection |
| `debug_test` | Systematic test debugging workflow |
| `complete_phase` | Phase completion checklist and guidance |
| `review_spec` | Comprehensive spec status review |

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
version = "0.2.0"
```

### LLM Configuration

Configure LLM providers for AI-powered features in the `[llm]` section:

```toml
[llm]
provider = "openai"           # "openai", "anthropic", or "local"
api_key = "sk-..."            # Optional: falls back to env var
model = "gpt-4"               # Optional: uses provider default
timeout = 30                  # Request timeout in seconds
base_url = "https://..."      # Optional: custom API endpoint
max_tokens = 1024             # Default max tokens for responses
temperature = 0.7             # Default temperature (0-2)
```

**LLM Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `FOUNDRY_MCP_LLM_PROVIDER` | LLM provider type | openai |
| `FOUNDRY_MCP_LLM_API_KEY` | Unified API key (overrides provider-specific) | None |
| `FOUNDRY_MCP_LLM_MODEL` | Model identifier | Provider default |
| `FOUNDRY_MCP_LLM_TIMEOUT` | Request timeout in seconds | 30 |
| `FOUNDRY_MCP_LLM_BASE_URL` | Custom API base URL | Provider default |
| `FOUNDRY_MCP_LLM_MAX_TOKENS` | Default max tokens | 1024 |
| `FOUNDRY_MCP_LLM_TEMPERATURE` | Default temperature | 0.7 |

Provider-specific API key fallbacks (used if `FOUNDRY_MCP_LLM_API_KEY` not set):
- `OPENAI_API_KEY` for OpenAI provider
- `ANTHROPIC_API_KEY` for Anthropic provider

### Workflow Configuration

Configure workflow execution modes in the `[workflow]` section:

```toml
[workflow]
mode = "single"               # "single", "autonomous", or "batch"
auto_validate = true          # Auto-validate after task completion
journal_enabled = true        # Enable task journaling
batch_size = 5                # Tasks per batch (batch mode)
context_threshold = 85        # Context usage threshold (%)
```

**Workflow Modes:**
- `single`: Execute one task at a time with user approval
- `autonomous`: Execute all phase tasks automatically until completion or blocker
- `batch`: Execute a batch of tasks, then pause for review

**Workflow Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `FOUNDRY_MCP_WORKFLOW_MODE` | Execution mode | single |
| `FOUNDRY_MCP_WORKFLOW_AUTO_VALIDATE` | Enable auto-validation | true |
| `FOUNDRY_MCP_WORKFLOW_JOURNAL_ENABLED` | Enable journaling | true |
| `FOUNDRY_MCP_WORKFLOW_BATCH_SIZE` | Batch size | 5 |
| `FOUNDRY_MCP_WORKFLOW_CONTEXT_THRESHOLD` | Context threshold (%) | 85 |

## Spec Directory Structure

foundry-mcp expects specs organized in status folders:

```
specs/
├── pending/      # New specs awaiting activation
├── active/       # Currently being worked on
├── completed/    # Finished specs
├── archived/     # Historical specs
└── templates/    # Custom spec templates (optional)
```

## Development

```bash
# Clone the repository
git clone https://github.com/tylerburleigh/foundry-mcp.git
cd foundry-mcp

# Install in development mode
pip install -e .

# Run tests
pytest

# Run the server
foundry-mcp
```

### Developer Documentation

| Document | Description |
|----------|-------------|
| [MCP Best Practices](docs/mcp_best_practices/README.md) | Comprehensive guide for building reliable MCP tools |
| [Response Schema Guide](docs/codebase_standards/mcp_response_schema.md) | Canonical response contract and helper usage |

## License

MIT
