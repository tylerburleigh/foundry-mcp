"""
Feature flags infrastructure for foundry-mcp.

Provides feature flag management with lifecycle states, percentage rollouts,
client-specific overrides, and testing utilities.

See docs/mcp_best_practices/14-feature-flags.md for guidance.

Example:
    from foundry_mcp.core.feature_flags import (
        FlagState, FeatureFlag, FeatureFlagRegistry, feature_flag, flag_override
    )

    # Define flags
    registry = FeatureFlagRegistry()
    registry.register(FeatureFlag(
        name="new_algorithm",
        description="Use improved processing algorithm",
        state=FlagState.BETA,
        default_enabled=False,
    ))

    # Gate feature with decorator
    @feature_flag("new_algorithm")
    def process_data(data: dict) -> dict:
        return improved_process(data)

    # Test with override
    with flag_override("new_algorithm", True):
        result = process_data({"input": "test"})
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class FlagState(str, Enum):
    """Lifecycle state for feature flags.

    Feature flags progress through states as they mature:

    EXPERIMENTAL: Early development, opt-in only, may change without notice, no SLA
    BETA: Feature-complete but not fully validated, default off, opt-in available, some SLA
    STABLE: Production-ready, default on, full SLA guarantees
    DEPRECATED: Being phased out, warns on use, will be removed after expiration

    Example:
        >>> flag = FeatureFlag(name="my_feature", state=FlagState.EXPERIMENTAL)
        >>> if flag.state == FlagState.DEPRECATED:
        ...     logger.warning("Feature is deprecated")
    """

    EXPERIMENTAL = "experimental"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"


@dataclass
class FeatureFlag:
    """Definition of a feature flag with lifecycle and rollout configuration.

    Attributes:
        name: Unique identifier for the flag (used in code and config)
        description: Human-readable description of what this flag controls
        state: Current lifecycle state (experimental/beta/stable/deprecated)
        default_enabled: Whether the flag is enabled by default
        created_at: When the flag was created (defaults to now)
        expires_at: Optional expiration date after which flag should be removed
        owner: Team or individual responsible for this flag
        percentage_rollout: Percentage of clients that should have flag enabled (0-100)
        allowed_clients: If non-empty, only these clients can access the flag
        blocked_clients: These clients are explicitly denied access
        dependencies: Other flags that must be enabled for this flag to be enabled
        metadata: Additional key-value metadata for the flag

    Example:
        >>> flag = FeatureFlag(
        ...     name="new_search",
        ...     description="Enable new search algorithm",
        ...     state=FlagState.BETA,
        ...     default_enabled=False,
        ...     percentage_rollout=25.0,
        ...     owner="search-team",
        ... )
        >>> flag.is_expired()
        False
    """

    name: str
    description: str
    state: FlagState
    default_enabled: bool
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: Optional[datetime] = None
    owner: str = ""
    percentage_rollout: float = 100.0
    allowed_clients: Set[str] = field(default_factory=set)
    blocked_clients: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the flag has passed its expiration date."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "default_enabled": self.default_enabled,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "owner": self.owner,
            "percentage_rollout": self.percentage_rollout,
            "allowed_clients": list(self.allowed_clients),
            "blocked_clients": list(self.blocked_clients),
            "dependencies": list(self.dependencies),
            "metadata": self.metadata,
        }


# Export all public symbols
__all__ = [
    "FlagState",
    "FeatureFlag",
]
