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


@dataclass
class ServerCapabilities:
    """
    Server capabilities for client negotiation.

    Clients call get_capabilities() to understand what features are supported
    before making assumptions about available functionality.

    Attributes:
        response_version: Response contract version (e.g., "response-v2")
        supports_streaming: Whether server supports streaming responses
        supports_batch: Whether server supports batch operations
        supports_pagination: Whether server supports cursor-based pagination
        max_batch_size: Maximum items in a batch request
        rate_limit_headers: Whether responses include rate limit headers
        supported_formats: List of supported response formats
        feature_flags_enabled: Whether feature flags are active
    """

    response_version: str = "response-v2"
    supports_streaming: bool = False
    supports_batch: bool = True
    supports_pagination: bool = True
    max_batch_size: int = 100
    rate_limit_headers: bool = True
    supported_formats: List[str] = field(default_factory=lambda: ["json"])
    feature_flags_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert capabilities to dict for response.

        Returns:
            Dict suitable for capability negotiation responses
        """
        return {
            "response_version": self.response_version,
            "streaming": self.supports_streaming,
            "batch_operations": self.supports_batch,
            "pagination": self.supports_pagination,
            "max_batch_size": self.max_batch_size,
            "rate_limit_headers": self.rate_limit_headers,
            "formats": self.supported_formats,
            "feature_flags": self.feature_flags_enabled,
        }


# Global capabilities instance
_capabilities: Optional[ServerCapabilities] = None


def get_capabilities() -> Dict[str, Any]:
    """
    Get server capabilities for client negotiation.

    Clients should call this to understand server features before making
    assumptions about available functionality.

    Returns:
        Dict with capabilities, server version, and API version
    """
    global _capabilities
    if _capabilities is None:
        _capabilities = ServerCapabilities()

    return {
        "schema_version": SCHEMA_VERSION,
        "capabilities": _capabilities.to_dict(),
        "server_version": "1.0.0",
        "api_version": "2024-11-01",
    }


def negotiate_capabilities(
    requested_version: Optional[str] = None,
    requested_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Negotiate capabilities with client.

    Args:
        requested_version: Desired response version
        requested_features: List of requested feature names

    Returns:
        Dict with negotiated capabilities and any warnings
    """
    global _capabilities
    if _capabilities is None:
        _capabilities = ServerCapabilities()

    negotiated: Dict[str, Any] = {}
    warnings: List[str] = []

    # Version negotiation
    if requested_version:
        if requested_version == _capabilities.response_version:
            negotiated["response_version"] = requested_version
        else:
            negotiated["response_version"] = _capabilities.response_version
            warnings.append(
                f"Requested version '{requested_version}' not supported, "
                f"using '{_capabilities.response_version}'"
            )

    # Feature negotiation
    available_features = {
        "streaming": _capabilities.supports_streaming,
        "batch": _capabilities.supports_batch,
        "pagination": _capabilities.supports_pagination,
        "rate_limit_headers": _capabilities.rate_limit_headers,
        "feature_flags": _capabilities.feature_flags_enabled,
    }

    if requested_features:
        negotiated["features"] = {}
        for feature in requested_features:
            if feature in available_features:
                negotiated["features"][feature] = available_features[feature]
            else:
                negotiated["features"][feature] = False
                warnings.append(f"Feature '{feature}' not recognized")

    return {
        "schema_version": SCHEMA_VERSION,
        "negotiated": negotiated,
        "warnings": warnings if warnings else None,
    }


def set_capabilities(capabilities: ServerCapabilities) -> None:
    """
    Set server capabilities (for testing or custom configuration).

    Args:
        capabilities: ServerCapabilities instance to use
    """
    global _capabilities
    _capabilities = capabilities


# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def deprecated_tool(
    replacement: str,
    removal_version: str,
) -> Callable[[F], F]:
    """
    Decorator to mark MCP tools as deprecated.

    Modifies the function's docstring to include deprecation notice and
    adds deprecation warning to response meta.warnings.

    Args:
        replacement: Name of the tool that replaces this one
        removal_version: Version in which this tool will be removed

    Returns:
        Decorated function that adds deprecation warnings to responses

    Example:
        >>> @mcp.tool()
        ... @deprecated_tool(replacement="get_user", removal_version="3.0.0")
        ... def fetch_user(user_id: str) -> dict:
        ...     '''Fetch user by ID (deprecated).'''
        ...     return get_user(user_id)
    """

    def decorator(func: F) -> F:
        original_doc = func.__doc__ or ""

        # Update docstring with deprecation notice
        func.__doc__ = f"""[DEPRECATED] {original_doc}

        ⚠️  This tool is deprecated and will be removed in version {removal_version}.
        Use '{replacement}' instead.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Add deprecation warning to response if it has meta
            if isinstance(result, dict):
                if "meta" not in result:
                    result["meta"] = {"version": "response-v2"}

                if "warnings" not in result["meta"]:
                    result["meta"]["warnings"] = []

                deprecation_warning = (
                    f"DEPRECATED: '{func.__name__}' will be removed in {removal_version}. "
                    f"Use '{replacement}' instead."
                )

                if deprecation_warning not in result["meta"]["warnings"]:
                    result["meta"]["warnings"].append(deprecation_warning)

            return result

        # Store deprecation metadata on the wrapper for introspection
        wrapper._deprecated = True  # type: ignore[attr-defined]
        wrapper._replacement = replacement  # type: ignore[attr-defined]
        wrapper._removal_version = removal_version  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def is_deprecated(func: Callable[..., Any]) -> bool:
    """
    Check if a function is marked as deprecated.

    Args:
        func: Function to check

    Returns:
        True if function has @deprecated_tool decorator
    """
    return getattr(func, "_deprecated", False)


def get_deprecation_info(func: Callable[..., Any]) -> Optional[Dict[str, str]]:
    """
    Get deprecation info for a deprecated function.

    Args:
        func: Function to check

    Returns:
        Dict with replacement and removal_version, or None if not deprecated
    """
    if not is_deprecated(func):
        return None

    return {
        "replacement": getattr(func, "_replacement", ""),
        "removal_version": getattr(func, "_removal_version", ""),
    }
