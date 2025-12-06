# 11. AI/LLM Integration Patterns

> Design MCP tools for effective use by AI agents and LLMs.

## Overview

MCP tools are primarily consumed by AI agents and LLMs. This creates unique requirements around response design, context management, and interaction patterns that differ from traditional APIs.

## Requirements

### MUST

- **Keep responses concise** - LLMs have context window limits
- **Return structured data** - easier to parse than prose
- **Document tool capabilities clearly** - LLMs rely on descriptions
- **Handle LLM-generated input safely** - may be malformed or adversarial

### SHOULD

- **Design for tool chaining** - outputs should work as inputs to other tools
- **Include confidence indicators** - when results are uncertain
- **Provide actionable next steps** - guide the LLM on what to do next
- **Support progressive disclosure** - summary first, details on request

### MAY

- **Include reasoning hints** - help LLMs understand results
- **Support batch operations** - reduce round trips
- **Provide tool suggestions** - recommend related tools

## Context Window Awareness

### Response Size Guidelines

| Response Type | Target Size | Rationale |
|--------------|-------------|-----------|
| Simple result | < 500 tokens | Leaves room for reasoning |
| List/search results | < 2000 tokens | Reasonable for iteration |
| Detailed report | < 5000 tokens | May need chunking |
| Large dataset | Paginate | Never dump entire dataset |

### Concise Response Design

```python
# Bad: Verbose prose response
{
    "data": {
        "message": "I have successfully completed the search operation. "
                   "After examining 1,523 files across 47 directories, "
                   "I found 3 matches that contain the search term you specified. "
                   "The files are listed below with their full paths and the "
                   "line numbers where matches were found..."
    }
}

# Good: Structured, scannable response
{
    "data": {
        "matches": [
            {"file": "src/auth.py", "line": 42, "snippet": "def authenticate(user):"},
            {"file": "src/api.py", "line": 156, "snippet": "auth = authenticate(request.user)"},
            {"file": "tests/test_auth.py", "line": 23, "snippet": "result = authenticate(mock_user)"}
        ],
        "stats": {
            "files_searched": 1523,
            "matches_found": 3
        }
    }
}
```

### Progressive Disclosure

```python
@mcp.tool()
def get_project_summary(project_id: str, detail_level: str = "summary") -> dict:
    """Get project information with configurable detail.

    Args:
        project_id: Project identifier
        detail_level: "summary" (default), "standard", or "full"
    """
    project = db.get_project(project_id)

    if detail_level == "summary":
        # Minimal response for initial exploration
        return asdict(success_response(data={
            "name": project.name,
            "status": project.status,
            "file_count": len(project.files),
            "hint": "Use detail_level='standard' for more info"
        }))

    elif detail_level == "standard":
        # Moderate detail for most use cases
        return asdict(success_response(data={
            "name": project.name,
            "status": project.status,
            "description": project.description,
            "files": [f.name for f in project.files[:20]],
            "has_more_files": len(project.files) > 20,
            "recent_activity": project.last_modified.isoformat()
        }))

    else:  # full
        # Complete details when specifically needed
        return asdict(success_response(data={
            "name": project.name,
            "status": project.status,
            "description": project.description,
            "files": [{"name": f.name, "size": f.size, "modified": f.modified.isoformat()}
                     for f in project.files],
            "contributors": [c.name for c in project.contributors],
            "settings": project.settings,
            "metadata": project.metadata
        }))
```

## Tool Chaining Design

### Output-as-Input Compatibility

Design tool outputs to be valid inputs for related tools:

```python
# Tool 1: Search returns IDs
@mcp.tool()
def search_files(query: str) -> dict:
    """Search for files matching query.

    Returns file_ids that can be passed to get_file_content.
    """
    results = search_engine.search(query)
    return asdict(success_response(data={
        "file_ids": [r.id for r in results],  # Directly usable
        "previews": [{"id": r.id, "name": r.name, "snippet": r.snippet}
                    for r in results]
    }))

# Tool 2: Accepts IDs from search
@mcp.tool()
def get_file_content(file_id: str) -> dict:
    """Get content of a file by ID.

    file_id: ID from search_files results
    """
    content = storage.get_file(file_id)
    return asdict(success_response(data={
        "file_id": file_id,
        "content": content,
        "lines": content.count('\n') + 1
    }))
```

### Action Suggestions

```python
@mcp.tool()
def analyze_code(file_path: str) -> dict:
    """Analyze code for issues."""
    issues = analyzer.analyze(file_path)

    return asdict(success_response(data={
        "issues": issues,
        "summary": f"Found {len(issues)} issues",
        "suggested_actions": [
            {
                "action": "fix_issue",
                "params": {"file": file_path, "issue_id": issues[0]["id"]},
                "description": f"Fix: {issues[0]['message']}"
            } if issues else None,
            {
                "action": "get_file_content",
                "params": {"file_id": file_path},
                "description": "View file contents"
            }
        ]
    }))
```

## Tool Descriptions for LLMs

### Good Tool Descriptions

```python
@mcp.tool()
def create_branch(
    branch_name: str,
    base_branch: str = "main",
    checkout: bool = True
) -> dict:
    """Create a new git branch.

    Creates a new branch from the specified base branch. Use this when
    starting work on a new feature or fix. The branch name should follow
    the pattern: type/description (e.g., feature/add-login, fix/null-check).

    Args:
        branch_name: Name for the new branch (e.g., "feature/user-auth")
        base_branch: Branch to create from (default: "main")
        checkout: Switch to the new branch after creation (default: True)

    Returns:
        Branch creation result with the new branch name and current HEAD.

    Example:
        create_branch("feature/add-login") -> Creates and checks out feature/add-login

    Related tools:
        - list_branches: See existing branches
        - switch_branch: Change to a different branch
        - commit_changes: Commit work on the current branch
    """
    ...
```

### Bad Tool Descriptions

```python
@mcp.tool()
def create_branch(branch_name: str, base: str = "main", co: bool = True) -> dict:
    """Create branch."""  # Too vague!
    ...

@mcp.tool()
def cb(bn: str, bb: str = "main") -> dict:
    """Branch creation utility function for VCS operations."""  # Cryptic names, jargon
    ...
```

## Confidence and Uncertainty

```python
@mcp.tool()
def classify_intent(text: str) -> dict:
    """Classify user intent from text.

    Returns classification with confidence score.
    """
    result = classifier.predict(text)

    response_data = {
        "intent": result.label,
        "confidence": result.score,  # 0.0 to 1.0
    }

    # Add guidance based on confidence
    if result.score < 0.5:
        response_data["warning"] = "Low confidence - consider asking user to clarify"
        response_data["alternatives"] = [
            {"intent": alt.label, "confidence": alt.score}
            for alt in result.alternatives[:3]
        ]
    elif result.score < 0.8:
        response_data["note"] = "Moderate confidence - verify if critical"

    return asdict(success_response(data=response_data))
```

## Handling LLM-Generated Input

### Input Validation for LLM Sources

```python
@mcp.tool()
def execute_query(query: str, table: str) -> dict:
    """Execute a database query.

    SAFETY: This tool validates inputs to prevent injection.
    Only SELECT queries on approved tables are allowed.
    """
    # Validate table against allowlist
    ALLOWED_TABLES = {"users", "products", "orders"}
    if table not in ALLOWED_TABLES:
        return asdict(error_response(
            error=f"Table '{table}' not allowed. Valid: {ALLOWED_TABLES}",
            data={"allowed_tables": list(ALLOWED_TABLES)}
        ))

    # Validate query structure
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return asdict(error_response(
            error="Only SELECT queries allowed",
            data={"error_code": "INVALID_QUERY_TYPE"}
        ))

    # Check for dangerous patterns
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", ";"]
    for keyword in dangerous:
        if keyword in query_upper:
            return asdict(error_response(
                error=f"Query contains disallowed keyword: {keyword}",
                data={"error_code": "DANGEROUS_QUERY"}
            ))

    # Execute with parameterization
    try:
        results = db.execute_read_only(query, table)
        return asdict(success_response(data={"results": results}))
    except Exception as e:
        return asdict(error_response(f"Query failed: {str(e)}"))
```

## Batch Operations

Reduce round trips for common multi-item operations:

```python
@mcp.tool()
def get_files_batch(file_ids: List[str], max_content_length: int = 1000) -> dict:
    """Get content of multiple files in one call.

    More efficient than multiple get_file calls for batch operations.

    Args:
        file_ids: List of file IDs (max 10)
        max_content_length: Truncate content at this length per file
    """
    if len(file_ids) > 10:
        return asdict(error_response(
            error="Maximum 10 files per batch",
            data={"max_batch_size": 10, "requested": len(file_ids)}
        ))

    results = []
    errors = []

    for file_id in file_ids:
        try:
            content = storage.get_file(file_id)
            if len(content) > max_content_length:
                content = content[:max_content_length]
                truncated = True
            else:
                truncated = False

            results.append({
                "file_id": file_id,
                "content": content,
                "truncated": truncated
            })
        except FileNotFoundError:
            errors.append({"file_id": file_id, "error": "not_found"})

    return asdict(success_response(
        data={"files": results, "errors": errors},
        warnings=[f"{len(errors)} files not found"] if errors else None
    ))
```

## Response Formatting for LLMs

### Structured vs Prose

```python
# Good: Structured data LLMs can reason about
{
    "data": {
        "analysis": {
            "sentiment": "negative",
            "score": -0.7,
            "keywords": ["frustrated", "broken", "unacceptable"],
            "suggested_response_tone": "empathetic"
        }
    }
}

# Avoid: Prose that's hard to extract data from
{
    "data": {
        "analysis": "The sentiment appears to be negative with a score of -0.7. "
                    "Key negative words include 'frustrated', 'broken', and "
                    "'unacceptable'. I would suggest responding with empathy."
    }
}
```

### Code Responses

```python
# Good: Clearly delimited code with metadata
{
    "data": {
        "language": "python",
        "code": "def hello():\n    print('Hello, world!')",
        "filename": "hello.py",
        "executable": True
    }
}

# Avoid: Code buried in prose
{
    "data": {
        "response": "Here's the code you requested:\n\ndef hello():\n    print('Hello')\n\nThis function will..."
    }
}
```

## Anti-Patterns

### Don't: Return Excessive Data

```python
# Bad: Dumps entire database
@mcp.tool()
def list_all_users() -> dict:
    users = db.get_all_users()  # Could be millions!
    return asdict(success_response(data={"users": users}))

# Good: Paginated with limits
@mcp.tool()
def list_users(limit: int = 20, cursor: str = None) -> dict:
    users = db.get_users(limit=min(limit, 100), cursor=cursor)
    return asdict(success_response(
        data={"users": users},
        pagination={"has_more": len(users) == limit}
    ))
```

### Don't: Require Multiple Calls for Simple Tasks

```python
# Bad: Requires 3 calls for common operation
get_user_id(email)  # Returns ID
get_user_profile(id)  # Returns profile
get_user_permissions(id)  # Returns permissions

# Good: Single call with options
get_user(email, include=["profile", "permissions"])
```

### Don't: Use Ambiguous Tool Names

```python
# Bad: Unclear what these do
@mcp.tool()
def process(data): ...

@mcp.tool()
def handle(item): ...

# Good: Clear, action-oriented names
@mcp.tool()
def validate_email_address(email): ...

@mcp.tool()
def convert_image_to_pdf(image_path): ...
```

## Related Documents

- [Security & Trust Boundaries](./08-security-trust-boundaries.md) - Input validation
- [Validation & Input Hygiene](./04-validation-input-hygiene.md) - Sanitization
- [Tool Discovery](./13-tool-discovery.md) - Tool descriptions

---

**Navigation:** [← Testing & Fixtures](./10-testing-fixtures.md) | [Index](./README.md) | [Next: Timeout & Resilience →](./12-timeout-resilience.md)
