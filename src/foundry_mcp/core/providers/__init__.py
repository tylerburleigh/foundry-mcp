"""
Provider abstractions for foundry-mcp.

This package provides pluggable LLM provider backends for CLI operations,
with support for capability negotiation, request/response normalization,
lifecycle hooks, availability detection, and registry management.

Example usage:
    from foundry_mcp.core.providers import (
        # Core types
        ProviderCapability,
        ProviderRequest,
        ProviderResult,
        ProviderContext,
        ProviderHooks,
        # Detection
        detect_provider_availability,
        get_provider_statuses,
        # Registry
        register_provider,
        resolve_provider,
        available_providers,
    )

    # Check provider availability
    if detect_provider_availability("gemini"):
        # Register and resolve a provider
        hooks = ProviderHooks()
        provider = resolve_provider("gemini", hooks=hooks)

        # Check if provider supports streaming
        if provider.supports(ProviderCapability.STREAMING):
            request = ProviderRequest(prompt="Hello", stream=True)
            result = provider.generate(request)
"""

from foundry_mcp.core.providers.base import (
    # Enums
    ProviderCapability,
    ProviderStatus,
    # Request/Response dataclasses
    ProviderRequest,
    ProviderResult,
    TokenUsage,
    StreamChunk,
    # Metadata dataclasses
    ModelDescriptor,
    ProviderMetadata,
    # Hooks
    ProviderHooks,
    StreamChunkCallback,
    BeforeExecuteHook,
    AfterResultHook,
    # Errors
    ProviderError,
    ProviderUnavailableError,
    ProviderExecutionError,
    ProviderTimeoutError,
    # ABC
    ProviderContext,
)

from foundry_mcp.core.providers.detectors import (
    ProviderDetector,
    register_detector,
    get_detector,
    detect_provider_availability,
    get_provider_statuses,
    list_detectors,
    reset_detectors,
)

from foundry_mcp.core.providers.registry import (
    # Types
    ProviderFactory,
    ProviderRegistration,
    AvailabilityCheck,
    MetadataResolver,
    LazyFactoryLoader,
    DependencyResolver,
    # Registration
    register_provider,
    register_lazy_provider,
    # Resolution
    available_providers,
    check_provider_available,
    resolve_provider,
    get_provider_metadata,
    describe_providers,
    # Dependency Injection
    set_dependency_resolver,
    # Testing
    reset_registry,
    get_registration,
)

__all__ = [
    # === Base Types (base.py) ===
    # Enums
    "ProviderCapability",
    "ProviderStatus",
    # Request/Response dataclasses
    "ProviderRequest",
    "ProviderResult",
    "TokenUsage",
    "StreamChunk",
    # Metadata dataclasses
    "ModelDescriptor",
    "ProviderMetadata",
    # Hooks
    "ProviderHooks",
    "StreamChunkCallback",
    "BeforeExecuteHook",
    "AfterResultHook",
    # Errors
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderExecutionError",
    "ProviderTimeoutError",
    # ABC
    "ProviderContext",
    # === Detection (detectors.py) ===
    "ProviderDetector",
    "register_detector",
    "get_detector",
    "detect_provider_availability",
    "get_provider_statuses",
    "list_detectors",
    "reset_detectors",
    # === Registry (registry.py) ===
    # Types
    "ProviderFactory",
    "ProviderRegistration",
    "AvailabilityCheck",
    "MetadataResolver",
    "LazyFactoryLoader",
    "DependencyResolver",
    # Registration
    "register_provider",
    "register_lazy_provider",
    # Resolution
    "available_providers",
    "check_provider_available",
    "resolve_provider",
    "get_provider_metadata",
    "describe_providers",
    # Dependency Injection
    "set_dependency_resolver",
    # Testing
    "reset_registry",
    "get_registration",
]
