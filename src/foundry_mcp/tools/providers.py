"""
Provider tools for foundry-mcp.

Exposes LLM provider management capabilities through MCP tools:
- Provider listing and status
- Provider execution
- Capability introspection

These tools bridge the provider abstraction layer to MCP clients,
enabling agents to discover, check, and invoke AI providers.
"""

import logging
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool

logger = logging.getLogger(__name__)


def register_provider_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register provider tools with the FastMCP server.

    Exposes MCP tools for:
    - provider-list: List all registered providers with availability status
    - provider-status: Get detailed status for a specific provider
    - provider-execute: Execute a prompt through a provider

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    # Import provider dependencies inside the function to avoid
    # circular imports and allow lazy loading
    from foundry_mcp.core.providers import (
        available_providers,
        describe_providers,
        check_provider_available,
        get_provider_metadata,
        get_provider_statuses,
        resolve_provider,
        ProviderHooks,
        ProviderRequest,
        ProviderCapability,
        ProviderUnavailableError,
        ProviderExecutionError,
        ProviderTimeoutError,
    )

    @canonical_tool(
        mcp,
        canonical_name="provider-list",
    )
    def provider_list(
        include_unavailable: bool = False,
    ) -> dict:
        """
        List all registered LLM providers with availability status.

        Returns registered providers sorted by priority, with availability
        status indicating which can currently be used. Useful for discovering
        what AI backends are configured.

        WHEN TO USE:
        - Discovering available AI providers
        - Checking which providers are configured
        - Before selecting a provider for execution

        Args:
            include_unavailable: Include providers that fail availability check
                                 (default: False)

        Returns:
            JSON object with:
            - providers: List of provider summaries (id, description, priority,
                        tags, available)
            - available_count: Number of available providers
            - total_count: Total registered providers
        """
        try:
            # Get full provider descriptions
            all_providers = describe_providers()

            # Filter if needed
            if not include_unavailable:
                providers = [p for p in all_providers if p.get("available", False)]
            else:
                providers = all_providers

            # Count statistics
            available_count = sum(
                1 for p in all_providers if p.get("available", False)
            )
            total_count = len(all_providers)

            return asdict(
                success_response(
                    data={
                        "providers": providers,
                        "available_count": available_count,
                        "total_count": total_count,
                    }
                )
            )

        except Exception as e:
            logger.exception("Error listing providers")
            return asdict(error_response(f"Failed to list providers: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="provider-status",
    )
    def provider_status(
        provider_id: str,
    ) -> dict:
        """
        Get detailed status for a specific provider.

        Returns availability, metadata, capabilities, and health status
        for a registered provider. Use this to check if a provider is
        ready for use and what features it supports.

        WHEN TO USE:
        - Checking if a specific provider is available
        - Understanding provider capabilities before use
        - Debugging provider configuration issues

        Args:
            provider_id: Provider identifier (e.g., "gemini", "codex",
                        "cursor-agent", "claude", "opencode")

        Returns:
            JSON object with:
            - provider_id: The queried provider ID
            - available: Whether provider can be used
            - metadata: Provider metadata (models, capabilities, docs)
            - capabilities: List of supported capability flags
            - health: Current health status from detector
        """
        try:
            if not provider_id:
                return asdict(
                    error_response(
                        "provider_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a provider_id (e.g., 'gemini', 'codex')",
                    )
                )

            # Check if provider is available
            is_available = check_provider_available(provider_id)

            # Get metadata if available
            metadata = get_provider_metadata(provider_id)
            metadata_dict = None
            capabilities_list = None

            if metadata:
                metadata_dict = {
                    "name": metadata.name,
                    "version": metadata.version,
                    "default_model": metadata.default_model,
                    "supported_models": [
                        {
                            "id": m.model_id,
                            "name": m.name,
                            "context_window": m.context_window,
                            "is_default": m.is_default,
                        }
                        for m in (metadata.supported_models or [])
                    ],
                    "documentation_url": metadata.documentation_url,
                    "tags": list(metadata.tags) if metadata.tags else [],
                }
                capabilities_list = [
                    cap.value for cap in (metadata.capabilities or [])
                ]

            # Get health status from detector
            statuses = get_provider_statuses()
            health = statuses.get(provider_id)
            health_dict = None
            if health:
                health_dict = {
                    "status": health.status.value,
                    "reason": health.reason,
                    "checked_at": (
                        health.checked_at.isoformat()
                        if health.checked_at
                        else None
                    ),
                }

            if not is_available and metadata is None and health is None:
                return asdict(
                    error_response(
                        f"Provider '{provider_id}' not found",
                        error_code="NOT_FOUND",
                        error_type="not_found",
                        remediation="Use provider-list to see available providers",
                    )
                )

            return asdict(
                success_response(
                    data={
                        "provider_id": provider_id,
                        "available": is_available,
                        "metadata": metadata_dict,
                        "capabilities": capabilities_list,
                        "health": health_dict,
                    }
                )
            )

        except Exception as e:
            logger.exception(f"Error getting provider status for {provider_id}")
            return asdict(error_response(f"Failed to get provider status: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="provider-execute",
    )
    def provider_execute(
        provider_id: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """
        Execute a prompt through a specified LLM provider.

        Sends a prompt to the specified provider and returns the result.
        Supports provider-specific model selection and generation parameters.

        WHEN TO USE:
        - Invoking an AI provider for generation
        - Executing prompts through external CLI tools
        - Multi-model consultation workflows

        LIMITATIONS:
        - Requires provider to be available and configured
        - Streaming not supported through this tool (returns complete result)
        - Some providers may have rate limits or require API keys

        Args:
            provider_id: Provider identifier (e.g., "gemini", "codex")
            prompt: The prompt text to send to the provider
            model: Optional model override (uses provider default if not specified)
            max_tokens: Maximum tokens in response (provider-specific default)
            temperature: Sampling temperature 0.0-2.0 (provider-specific default)
            timeout: Request timeout in seconds (default: 300)

        Returns:
            JSON object with:
            - provider_id: Provider that handled the request
            - model: Model used for generation
            - content: Generated text response
            - token_usage: Token counts (prompt, completion, total) if available
            - finish_reason: Why generation stopped (if available)
        """
        try:
            # Validate required parameters
            if not provider_id:
                return asdict(
                    error_response(
                        "provider_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a provider_id (e.g., 'gemini', 'codex')",
                    )
                )

            if not prompt or not prompt.strip():
                return asdict(
                    error_response(
                        "prompt is required and cannot be empty",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a non-empty prompt string",
                    )
                )

            # Check availability before attempting resolution
            if not check_provider_available(provider_id):
                return asdict(
                    error_response(
                        f"Provider '{provider_id}' is not available",
                        error_code="UNAVAILABLE",
                        error_type="unavailable",
                        remediation=(
                            "Check provider configuration and availability. "
                            "Use provider-list to see available providers."
                        ),
                    )
                )

            # Create hooks (no streaming for MCP tool invocation)
            hooks = ProviderHooks()

            # Resolve provider instance
            provider = resolve_provider(
                provider_id,
                hooks=hooks,
                model=model,
            )

            # Build request
            request = ProviderRequest(
                prompt=prompt.strip(),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout or 300,
                stream=False,  # MCP tools don't support streaming
            )

            # Execute
            result = provider.generate(request)

            # Build response
            response_data = {
                "provider_id": provider_id,
                "model": result.model or model or "default",
                "content": result.content,
                "finish_reason": result.finish_reason,
            }

            # Include token usage if available
            if result.usage:
                response_data["token_usage"] = {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                }

            return asdict(success_response(data=response_data))

        except ProviderUnavailableError as e:
            logger.warning(f"Provider unavailable: {e}")
            return asdict(
                error_response(
                    str(e),
                    error_code="UNAVAILABLE",
                    error_type="unavailable",
                    data={"provider": e.provider} if e.provider else None,
                    remediation="Check provider configuration and availability",
                )
            )

        except ProviderTimeoutError as e:
            logger.warning(f"Provider timeout: {e}")
            return asdict(
                error_response(
                    str(e),
                    error_code="TIMEOUT",
                    error_type="unavailable",
                    data={"provider": e.provider} if e.provider else None,
                    remediation="Try again with a shorter prompt or increased timeout",
                )
            )

        except ProviderExecutionError as e:
            logger.warning(f"Provider execution error: {e}")
            return asdict(
                error_response(
                    str(e),
                    error_code="EXECUTION_ERROR",
                    error_type="internal",
                    data={"provider": e.provider} if e.provider else None,
                    remediation="Check provider logs and try again",
                )
            )

        except Exception as e:
            logger.exception(f"Error executing provider {provider_id}")
            return asdict(error_response(f"Failed to execute provider: {e}"))

    logger.debug("Provider tools registered")
