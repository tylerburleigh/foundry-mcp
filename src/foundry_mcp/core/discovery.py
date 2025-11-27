"""
Tool metadata and discovery module for foundry-mcp.

Provides dataclasses and utilities for tool registration, discovery,
capability negotiation, and deprecation handling per MCP best practices
(docs/mcp_best_practices/13-tool-discovery.md).
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Schema version for discovery responses
SCHEMA_VERSION = "1.0.0"


class ParameterType(str, Enum):
    """Types for tool parameters in JSON Schema."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ParameterMetadata:
    """
    Metadata for a tool parameter.

    Attributes:
        name: Parameter name
        type: Parameter type (string, integer, boolean, etc.)
        description: Human-readable description
        required: Whether parameter is required
        default: Default value if not provided
        constraints: JSON Schema constraints (pattern, min, max, enum, etc.)
        examples: Example values for documentation
    """

    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)


@dataclass
class ToolMetadata:
    """
    Comprehensive metadata for an MCP tool.

    Used for tool registration, discovery, and documentation generation.
    Supports categorization, versioning, deprecation, and rate limiting.

    Attributes:
        name: Unique tool identifier
        description: Human-readable description (supports markdown)
        parameters: List of parameter metadata
        category: Tool category for grouping (e.g., "files", "users", "code")
        version: Semantic version of the tool interface
        deprecated: Whether tool is deprecated
        deprecation_message: Message explaining deprecation and migration path
        rate_limit: Rate limit description (e.g., "100/minute")
        examples: List of usage examples with input/output
        related_tools: Names of related tools
        tags: Semantic tags for filtering
    """

    name: str
    description: str
    parameters: List[ParameterMetadata] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0.0"
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    rate_limit: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    related_tools: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Generate JSON Schema from tool metadata.

        Returns:
            JSON Schema dict for tool input validation
        """
        type_mapping = {
            ParameterType.STRING: "string",
            ParameterType.INTEGER: "integer",
            ParameterType.NUMBER: "number",
            ParameterType.BOOLEAN: "boolean",
            ParameterType.ARRAY: "array",
            ParameterType.OBJECT: "object",
        }

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": type_mapping[param.type],
                "description": param.description,
            }

            if param.default is not None:
                prop["default"] = param.default

            if param.examples:
                prop["examples"] = param.examples

            # Add constraints
            prop.update(param.constraints)

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.name,
            "description": self.description,
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def to_summary(self) -> Dict[str, Any]:
        """
        Generate summary for tool listing.

        Returns:
            Compact dict for list_tools responses
        """
        desc = self.description
        if len(desc) > 200:
            desc = desc[:197] + "..."

        return {
            "name": self.name,
            "description": desc,
            "category": self.category,
            "version": self.version,
            "deprecated": self.deprecated,
            "tags": self.tags,
        }

    def to_detailed(self) -> Dict[str, Any]:
        """
        Generate detailed info for get_tool_schema responses.

        Returns:
            Full dict including schema, examples, and related tools
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "schema": self.to_json_schema(),
            "examples": self.examples,
            "related_tools": self.related_tools,
            "tags": self.tags,
            "deprecated": self.deprecated,
        }

        if self.rate_limit:
            result["rate_limit"] = self.rate_limit

        if self.deprecation_message:
            result["deprecation_message"] = self.deprecation_message

        return result


class ToolRegistry:
    """
    Central registry for MCP tool metadata.

    Provides tool registration, discovery, and filtering capabilities.
    Used by MCP servers to expose tool information to clients.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(ToolMetadata(
        ...     name="get_user",
        ...     description="Get user by ID",
        ...     category="users",
        ... ))
        >>> tools = registry.list_tools(category="users")
    """

    def __init__(self) -> None:
        """Initialize empty tool registry."""
        self._tools: Dict[str, ToolMetadata] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: ToolMetadata) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool metadata to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool

        # Update category index
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        self._categories[tool.category].append(tool.name)

    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._tools:
            return False

        tool = self._tools.pop(name)
        if tool.category in self._categories:
            self._categories[tool.category].remove(name)
            if not self._categories[tool.category]:
                del self._categories[tool.category]

        return True

    def get(self, name: str) -> Optional[ToolMetadata]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            ToolMetadata if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        include_deprecated: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List available tools with filtering.

        Args:
            category: Filter by category
            tag: Filter by tag
            include_deprecated: Include deprecated tools (default: False)

        Returns:
            List of tool summaries
        """
        tools = list(self._tools.values())

        # Apply filters
        if category:
            tools = [t for t in tools if t.category == category]

        if tag:
            tools = [t for t in tools if tag in t.tags]

        if not include_deprecated:
            tools = [t for t in tools if not t.deprecated]

        return [t.to_summary() for t in tools]

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed schema for a specific tool.

        Args:
            name: Tool name

        Returns:
            Detailed tool info including schema, or None if not found
        """
        tool = self._tools.get(name)
        if tool is None:
            return None
        return tool.to_detailed()

    def list_categories(self) -> List[Dict[str, Any]]:
        """
        List tool categories with descriptions.

        Returns:
            List of categories with tool counts
        """
        result = []
        for category, tool_names in sorted(self._categories.items()):
            # Filter out deprecated tools from count
            active_count = sum(
                1
                for name in tool_names
                if name in self._tools and not self._tools[name].deprecated
            )

            result.append(
                {
                    "name": category,
                    "tool_count": active_count,
                    "tools": tool_names,
                }
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with total counts and breakdown
        """
        total = len(self._tools)
        deprecated = sum(1 for t in self._tools.values() if t.deprecated)

        return {
            "total_tools": total,
            "active_tools": total - deprecated,
            "deprecated_tools": deprecated,
            "categories": len(self._categories),
        }


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
