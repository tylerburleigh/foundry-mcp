"""
AI Consultation Layer for foundry-mcp.

This module provides a unified interface for AI-assisted operations including
document generation, plan review, and fidelity checking. It integrates with
the provider registry to support multiple LLM backends while providing
caching, timeout handling, and consistent result structures.

Design Principles:
    - Workflow-specific prompt templates (doc_generation, plan_review, fidelity_review)
    - Provider-agnostic orchestration via the provider registry
    - Filesystem-based caching for consultation results
    - Consistent result structures across all workflows
    - Graceful degradation when providers are unavailable

Example Usage:
    from foundry_mcp.core.ai_consultation import (
        ConsultationOrchestrator,
        ConsultationRequest,
        ConsultationWorkflow,
    )
    from foundry_mcp.core.providers import ProviderHooks

    orchestrator = ConsultationOrchestrator()

    # Check availability
    if orchestrator.is_available():
        request = ConsultationRequest(
            workflow=ConsultationWorkflow.DOC_GENERATION,
            prompt_id="analyze_module",
            context={"file_path": "src/main.py", "content": "..."},
            provider_id="gemini",
        )
        result = orchestrator.consult(request)
        if result.content:
            print(result.content)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from foundry_mcp.core.providers import (
    ProviderHooks,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderUnavailableError,
    available_providers,
    check_provider_available,
    resolve_provider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Workflow Types
# =============================================================================


class ConsultationWorkflow(str, Enum):
    """
    Supported AI consultation workflows.

    Each workflow corresponds to a category of prompt templates and
    determines cache partitioning and result handling.

    Values:
        DOC_GENERATION: Generate documentation from code analysis
        PLAN_REVIEW: Review and critique SDD specifications
        FIDELITY_REVIEW: Compare implementation against specifications
    """

    DOC_GENERATION = "doc_generation"
    PLAN_REVIEW = "plan_review"
    FIDELITY_REVIEW = "fidelity_review"


# =============================================================================
# Request/Response Dataclasses
# =============================================================================


@dataclass(frozen=True)
class ConsultationRequest:
    """
    Request payload for AI consultation.

    Encapsulates all parameters needed to execute a consultation workflow,
    including prompt selection, context data, and provider preferences.

    Attributes:
        workflow: The consultation workflow type
        prompt_id: Identifier for the prompt template within the workflow
        context: Structured context data to inject into the prompt
        provider_id: Optional preferred provider (uses first available if None)
        model: Optional model override for the provider
        cache_key: Optional explicit cache key (auto-generated if None)
        timeout: Request timeout in seconds (default: 120)
        temperature: Sampling temperature (default: provider default)
        max_tokens: Maximum output tokens (default: provider default)
        system_prompt_override: Optional system prompt override
    """

    workflow: ConsultationWorkflow
    prompt_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    provider_id: Optional[str] = None
    model: Optional[str] = None
    cache_key: Optional[str] = None
    timeout: float = 120.0
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt_override: Optional[str] = None


@dataclass
class ConsultationResult:
    """
    Result of an AI consultation.

    Provides a consistent structure for consultation outcomes across all
    workflows and providers, including metadata for debugging and analytics.

    Attributes:
        workflow: The workflow that produced this result
        content: The generated content (may be empty on failure)
        provider_id: Provider that handled the request
        model_used: Fully-qualified model identifier
        tokens: Token usage if reported by provider
        duration_ms: Total consultation duration in milliseconds
        cache_hit: Whether result was served from cache
        raw_payload: Provider-specific metadata and debug info
        warnings: Non-fatal issues encountered during consultation
        error: Error message if consultation failed
    """

    workflow: ConsultationWorkflow
    content: str
    provider_id: str
    model_used: str
    tokens: Dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    cache_hit: bool = False
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True if consultation succeeded (has content, no error)."""
        return bool(self.content) and self.error is None


# =============================================================================
# Cache Implementation
# =============================================================================


class ResultCache:
    """
    Filesystem-based cache for consultation results.

    Provides persistent caching of AI consultation results to reduce
    redundant API calls and improve response times for repeated queries.

    Cache Structure:
        .cache/foundry-mcp/consultations/{workflow}/{key}.json

    Each cached entry contains:
        - content: The consultation result
        - provider_id: Provider that generated the result
        - model_used: Model identifier
        - tokens: Token usage
        - timestamp: Cache entry creation time
        - ttl: Time-to-live in seconds

    Attributes:
        base_dir: Root directory for cache storage
        default_ttl: Default time-to-live in seconds (default: 3600 = 1 hour)
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        default_ttl: int = 3600,
    ):
        """
        Initialize the result cache.

        Args:
            base_dir: Root directory for cache (default: .cache/foundry-mcp/consultations)
            default_ttl: Default TTL in seconds (default: 3600)
        """
        if base_dir is None:
            base_dir = Path.cwd() / ".cache" / "foundry-mcp" / "consultations"
        self.base_dir = base_dir
        self.default_ttl = default_ttl

    def _get_cache_path(self, workflow: ConsultationWorkflow, key: str) -> Path:
        """Return the cache file path for a workflow and key."""
        # Sanitize key to be filesystem-safe
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.base_dir / workflow.value / f"{safe_key}.json"

    def get(
        self,
        workflow: ConsultationWorkflow,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached result.

        Args:
            workflow: The consultation workflow
            key: The cache key

        Returns:
            Cached data dict if found and not expired, None otherwise
        """
        cache_path = self._get_cache_path(workflow, key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check TTL
            timestamp = data.get("timestamp", 0)
            ttl = data.get("ttl", self.default_ttl)
            if time.time() - timestamp > ttl:
                # Expired - remove file
                cache_path.unlink(missing_ok=True)
                return None

            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read cache entry %s: %s", cache_path, exc)
            return None

    def set(
        self,
        workflow: ConsultationWorkflow,
        key: str,
        result: ConsultationResult,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a consultation result in the cache.

        Args:
            workflow: The consultation workflow
            key: The cache key
            result: The consultation result to cache
            ttl: Time-to-live in seconds (default: default_ttl)
        """
        cache_path = self._get_cache_path(workflow, key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "content": result.content,
            "provider_id": result.provider_id,
            "model_used": result.model_used,
            "tokens": result.tokens,
            "timestamp": time.time(),
            "ttl": ttl if ttl is not None else self.default_ttl,
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to write cache entry %s: %s", cache_path, exc)

    def invalidate(
        self,
        workflow: Optional[ConsultationWorkflow] = None,
        key: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            workflow: If provided, only invalidate entries for this workflow
            key: If provided (with workflow), only invalidate this specific entry

        Returns:
            Number of entries invalidated
        """
        count = 0

        if workflow is not None and key is not None:
            # Invalidate specific entry
            cache_path = self._get_cache_path(workflow, key)
            if cache_path.exists():
                cache_path.unlink()
                count = 1
        elif workflow is not None:
            # Invalidate all entries for workflow
            workflow_dir = self.base_dir / workflow.value
            if workflow_dir.exists():
                for cache_file in workflow_dir.glob("*.json"):
                    cache_file.unlink()
                    count += 1
        else:
            # Invalidate all entries
            for workflow_enum in ConsultationWorkflow:
                workflow_dir = self.base_dir / workflow_enum.value
                if workflow_dir.exists():
                    for cache_file in workflow_dir.glob("*.json"):
                        cache_file.unlink()
                        count += 1

        return count

    def stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with entry counts per workflow and total size
        """
        stats: Dict[str, Any] = {
            "total_entries": 0,
            "total_size_bytes": 0,
            "by_workflow": {},
        }

        for workflow in ConsultationWorkflow:
            workflow_dir = self.base_dir / workflow.value
            if workflow_dir.exists():
                entries = list(workflow_dir.glob("*.json"))
                size = sum(f.stat().st_size for f in entries if f.exists())
                stats["by_workflow"][workflow.value] = {
                    "entries": len(entries),
                    "size_bytes": size,
                }
                stats["total_entries"] += len(entries)
                stats["total_size_bytes"] += size
            else:
                stats["by_workflow"][workflow.value] = {
                    "entries": 0,
                    "size_bytes": 0,
                }

        return stats


# =============================================================================
# Consultation Orchestrator
# =============================================================================


class ConsultationOrchestrator:
    """
    Central orchestrator for AI consultation workflows.

    Coordinates between prompt templates, the provider registry, and
    the result cache to execute consultation requests. Handles provider
    selection, timeout management, and error handling.

    Attributes:
        cache: ResultCache instance for caching results
        preferred_providers: Ordered list of preferred provider IDs
        default_timeout: Default timeout in seconds

    Example:
        orchestrator = ConsultationOrchestrator()

        if orchestrator.is_available():
            request = ConsultationRequest(
                workflow=ConsultationWorkflow.DOC_GENERATION,
                prompt_id="analyze_module",
                context={"content": "def foo(): pass"},
            )
            result = orchestrator.consult(request)
    """

    def __init__(
        self,
        cache: Optional[ResultCache] = None,
        preferred_providers: Optional[Sequence[str]] = None,
        default_timeout: float = 120.0,
    ):
        """
        Initialize the consultation orchestrator.

        Args:
            cache: ResultCache instance (creates default if None)
            preferred_providers: Ordered list of preferred provider IDs
            default_timeout: Default timeout in seconds
        """
        self.cache = cache or ResultCache()
        self.preferred_providers = list(preferred_providers) if preferred_providers else []
        self.default_timeout = default_timeout

    def is_available(self, provider_id: Optional[str] = None) -> bool:
        """
        Check if consultation services are available.

        Args:
            provider_id: Check specific provider, or any available if None

        Returns:
            True if at least one provider is available
        """
        if provider_id:
            return check_provider_available(provider_id)

        # Check preferred providers first
        for prov_id in self.preferred_providers:
            if check_provider_available(prov_id):
                return True

        # Fall back to any available provider
        return len(available_providers()) > 0

    def get_available_providers(self) -> List[str]:
        """
        Return list of available provider IDs.

        Preferred providers are listed first (if available), followed by
        other available providers.

        Returns:
            List of available provider IDs
        """
        available = set(available_providers())
        result = []

        # Add preferred providers that are available
        for prov_id in self.preferred_providers:
            if prov_id in available:
                result.append(prov_id)
                available.discard(prov_id)

        # Add remaining available providers
        result.extend(sorted(available))
        return result

    def _select_provider(self, request: ConsultationRequest) -> str:
        """
        Select the provider to use for a request.

        Args:
            request: The consultation request

        Returns:
            Provider ID to use

        Raises:
            ProviderUnavailableError: If no providers are available
        """
        # Explicit provider requested
        if request.provider_id:
            if check_provider_available(request.provider_id):
                return request.provider_id
            raise ProviderUnavailableError(
                f"Requested provider '{request.provider_id}' is not available",
                provider=request.provider_id,
            )

        # Try preferred providers
        for prov_id in self.preferred_providers:
            if check_provider_available(prov_id):
                return prov_id

        # Fall back to first available
        providers = available_providers()
        if providers:
            return providers[0]

        raise ProviderUnavailableError(
            "No AI providers are currently available",
            provider=None,
        )

    def _generate_cache_key(self, request: ConsultationRequest) -> str:
        """
        Generate a cache key for a consultation request.

        Args:
            request: The consultation request

        Returns:
            Cache key string
        """
        if request.cache_key:
            return request.cache_key

        # Build a deterministic key from request parameters
        key_parts = [
            request.prompt_id,
            json.dumps(request.context, sort_keys=True),
            request.model or "default",
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def _build_prompt(self, request: ConsultationRequest) -> str:
        """
        Build the full prompt from template and context.

        This method delegates to workflow-specific prompt builders.

        Args:
            request: The consultation request

        Returns:
            The rendered prompt string
        """
        # Import prompt builders lazily to avoid circular imports
        from foundry_mcp.core.prompts import get_prompt_builder

        builder = get_prompt_builder(request.workflow)
        return builder.build(request.prompt_id, request.context)

    def consult(
        self,
        request: ConsultationRequest,
        *,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
    ) -> ConsultationResult:
        """
        Execute a consultation request.

        Args:
            request: The consultation request
            use_cache: Whether to use cached results (default: True)
            cache_ttl: Cache TTL override in seconds

        Returns:
            ConsultationResult with the outcome
        """
        start_time = time.time()
        warnings: List[str] = []

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Check cache
        if use_cache:
            cached = self.cache.get(request.workflow, cache_key)
            if cached:
                duration_ms = (time.time() - start_time) * 1000
                return ConsultationResult(
                    workflow=request.workflow,
                    content=cached.get("content", ""),
                    provider_id=cached.get("provider_id", "cached"),
                    model_used=cached.get("model_used", "cached"),
                    tokens=cached.get("tokens", {}),
                    duration_ms=duration_ms,
                    cache_hit=True,
                )

        # Select provider
        try:
            provider_id = self._select_provider(request)
        except ProviderUnavailableError as exc:
            duration_ms = (time.time() - start_time) * 1000
            return ConsultationResult(
                workflow=request.workflow,
                content="",
                provider_id="none",
                model_used="none",
                duration_ms=duration_ms,
                error=str(exc),
            )

        # Build prompt
        try:
            prompt = self._build_prompt(request)
        except Exception as exc:  # noqa: BLE001 - wrap prompt build errors
            duration_ms = (time.time() - start_time) * 1000
            return ConsultationResult(
                workflow=request.workflow,
                content="",
                provider_id=provider_id,
                model_used="none",
                duration_ms=duration_ms,
                error=f"Failed to build prompt: {exc}",
            )

        # Execute via provider
        hooks = ProviderHooks()
        try:
            provider = resolve_provider(provider_id, hooks=hooks, model=request.model)
            provider_request = ProviderRequest(
                prompt=prompt,
                system_prompt=request.system_prompt_override,
                model=request.model,
                timeout=request.timeout or self.default_timeout,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            provider_result: ProviderResult = provider.generate(provider_request)
        except ProviderUnavailableError as exc:
            duration_ms = (time.time() - start_time) * 1000
            return ConsultationResult(
                workflow=request.workflow,
                content="",
                provider_id=provider_id,
                model_used="none",
                duration_ms=duration_ms,
                error=f"Provider unavailable: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - wrap provider errors
            duration_ms = (time.time() - start_time) * 1000
            return ConsultationResult(
                workflow=request.workflow,
                content="",
                provider_id=provider_id,
                model_used="none",
                duration_ms=duration_ms,
                error=f"Provider execution failed: {exc}",
            )

        # Build result
        duration_ms = (time.time() - start_time) * 1000
        tokens = {
            "input_tokens": provider_result.tokens.input_tokens,
            "output_tokens": provider_result.tokens.output_tokens,
            "total_tokens": provider_result.tokens.total_tokens,
        }

        # Handle non-success status
        error_msg = None
        if provider_result.status != ProviderStatus.SUCCESS:
            error_msg = f"Provider returned status: {provider_result.status.value}"
            if provider_result.stderr:
                error_msg += f" - {provider_result.stderr}"

        result = ConsultationResult(
            workflow=request.workflow,
            content=provider_result.content,
            provider_id=provider_result.provider_id,
            model_used=provider_result.model_used,
            tokens=tokens,
            duration_ms=duration_ms,
            cache_hit=False,
            raw_payload=provider_result.raw_payload,
            warnings=warnings,
            error=error_msg,
        )

        # Cache successful results
        if result.success and use_cache:
            self.cache.set(request.workflow, cache_key, result, ttl=cache_ttl)

        return result

    def consult_multiple(
        self,
        requests: Sequence[ConsultationRequest],
        *,
        use_cache: bool = True,
    ) -> List[ConsultationResult]:
        """
        Execute multiple consultation requests sequentially.

        Args:
            requests: Sequence of consultation requests
            use_cache: Whether to use cached results

        Returns:
            List of ConsultationResult objects in the same order as requests
        """
        return [self.consult(req, use_cache=use_cache) for req in requests]


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Workflow types
    "ConsultationWorkflow",
    # Request/Response
    "ConsultationRequest",
    "ConsultationResult",
    # Cache
    "ResultCache",
    # Orchestrator
    "ConsultationOrchestrator",
]
