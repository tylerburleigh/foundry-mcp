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
from enum import Enum

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


# Export all public symbols
__all__ = [
    "FlagState",
]
