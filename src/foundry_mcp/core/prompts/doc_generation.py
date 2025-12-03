"""
Prompt templates for documentation generation workflow.

This module provides prompts for analyzing code and generating
various types of documentation including module summaries,
API documentation, and architectural overviews.

Prompt IDs:
    - analyze_module: Analyze a Python module and describe its purpose
    - analyze_function: Document a specific function with examples
    - analyze_class: Document a class with its methods and attributes
    - generate_overview: Generate architectural overview for a package
    - summarize_changes: Summarize changes between code versions
"""

from __future__ import annotations

from typing import Any, Dict

from foundry_mcp.core.prompts import PromptBuilder


# =============================================================================
# Prompt Templates
# =============================================================================


ANALYZE_MODULE_TEMPLATE = """Analyze the following Python module and provide a comprehensive documentation summary.

## Module Information
- File Path: {file_path}
- Module Name: {module_name}

## Source Code
```python
{content}
```

## Analysis Requirements
1. **Purpose**: Describe the primary purpose and responsibility of this module
2. **Public API**: List and describe all public functions, classes, and constants
3. **Dependencies**: Note any notable imports and their usage
4. **Design Patterns**: Identify any design patterns or architectural decisions
5. **Usage Examples**: Provide 1-2 concise usage examples if applicable

## Output Format
Provide a structured markdown response with clear sections for each requirement above.
Keep the documentation concise but complete - aim for clarity over verbosity.
"""


ANALYZE_FUNCTION_TEMPLATE = """Document the following function with detailed information suitable for API documentation.

## Function Information
- Function Name: {function_name}
- File Path: {file_path}
- Module: {module_name}

## Source Code
```python
{content}
```

## Context (if available)
{context_info}

## Documentation Requirements
1. **Summary**: One-sentence description of what the function does
2. **Parameters**: Document each parameter with type, description, and constraints
3. **Returns**: Document the return value with type and description
4. **Raises**: List exceptions that may be raised
5. **Examples**: Provide 1-2 usage examples showing common use cases
6. **Notes**: Any important implementation details or caveats

## Output Format
Return the documentation in Google-style docstring format, ready to be inserted into the source code.
"""


ANALYZE_CLASS_TEMPLATE = """Document the following class with comprehensive API documentation.

## Class Information
- Class Name: {class_name}
- File Path: {file_path}
- Module: {module_name}
- Base Classes: {base_classes}

## Source Code
```python
{content}
```

## Documentation Requirements
1. **Summary**: Describe the class purpose and when to use it
2. **Attributes**: Document all public attributes
3. **Methods**: List and describe all public methods with signatures
4. **Constructor**: Document __init__ parameters
5. **Usage Example**: Show how to instantiate and use the class
6. **Related Classes**: Note any important relationships or inheritance

## Output Format
Provide markdown documentation suitable for a developer guide.
Include a class diagram in ASCII if the class has complex relationships.
"""


GENERATE_OVERVIEW_TEMPLATE = """Generate an architectural overview for the following package/module collection.

## Package Information
- Package Path: {package_path}
- Package Name: {package_name}

## Module Summary
{module_summary}

## File Structure
{file_structure}

## Requirements
1. **Architecture Overview**: Describe the high-level architecture and design
2. **Component Relationships**: How do the modules interact?
3. **Entry Points**: What are the main entry points for users?
4. **Extension Points**: How can the package be extended?
5. **Dependencies**: External dependencies and their roles

## Output Format
Provide a markdown document suitable for a README or architecture documentation.
Include ASCII diagrams where helpful for understanding relationships.
"""


SUMMARIZE_CHANGES_TEMPLATE = """Summarize the changes between two versions of the code.

## Change Information
- File Path: {file_path}
- Change Type: {change_type}

## Previous Version
```python
{old_content}
```

## Current Version
```python
{new_content}
```

## Summary Requirements
1. **What Changed**: Describe the key changes made
2. **Why It Matters**: Explain the impact of these changes
3. **Breaking Changes**: Note any breaking changes to the public API
4. **Migration Notes**: If applicable, how to update calling code

## Output Format
Provide a concise changelog-style summary suitable for release notes.
"""


# =============================================================================
# Template Registry
# =============================================================================


TEMPLATES = {
    "analyze_module": ANALYZE_MODULE_TEMPLATE,
    "analyze_function": ANALYZE_FUNCTION_TEMPLATE,
    "analyze_class": ANALYZE_CLASS_TEMPLATE,
    "generate_overview": GENERATE_OVERVIEW_TEMPLATE,
    "summarize_changes": SUMMARIZE_CHANGES_TEMPLATE,
}


# =============================================================================
# Prompt Builder Implementation
# =============================================================================


class DocGenerationPromptBuilder(PromptBuilder):
    """
    Prompt builder for documentation generation workflow.

    Provides templates for analyzing code and generating various types
    of technical documentation.
    """

    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a documentation generation prompt.

        Args:
            prompt_id: One of: analyze_module, analyze_function, analyze_class,
                      generate_overview, summarize_changes
            context: Template context variables

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt_id is not recognized
        """
        template = TEMPLATES.get(prompt_id)
        if template is None:
            available = ", ".join(sorted(TEMPLATES.keys()))
            raise ValueError(
                f"Unknown prompt_id '{prompt_id}'. Available: {available}"
            )

        # Provide safe defaults for optional context keys
        safe_context = {
            "file_path": context.get("file_path", "unknown"),
            "module_name": context.get("module_name", "unknown"),
            "content": context.get("content", ""),
            "function_name": context.get("function_name", "unknown"),
            "class_name": context.get("class_name", "unknown"),
            "base_classes": context.get("base_classes", "object"),
            "context_info": context.get("context_info", "No additional context provided."),
            "package_path": context.get("package_path", "unknown"),
            "package_name": context.get("package_name", "unknown"),
            "module_summary": context.get("module_summary", "Not provided"),
            "file_structure": context.get("file_structure", "Not provided"),
            "change_type": context.get("change_type", "modification"),
            "old_content": context.get("old_content", ""),
            "new_content": context.get("new_content", ""),
        }

        try:
            return template.format(**safe_context)
        except KeyError as exc:
            raise ValueError(f"Missing required context key: {exc}") from exc

    def list_prompts(self) -> list[str]:
        """Return available prompt IDs for documentation generation."""
        return sorted(TEMPLATES.keys())


__all__ = [
    "DocGenerationPromptBuilder",
    "TEMPLATES",
]
