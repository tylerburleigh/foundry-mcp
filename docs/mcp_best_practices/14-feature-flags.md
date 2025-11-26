# 14. Feature Flags & Gradual Rollouts

> Control feature exposure and safely deploy changes with feature flags.

## Overview

Feature flags enable controlled rollouts, A/B testing, and quick rollbacks without deployments. For MCP tools, flags can gate new tool versions, experimental features, or schema changes.

## Requirements

### MUST

- **Document flag lifecycle** - experimental → beta → stable → deprecated
- **Provide flag status in capabilities** - clients must know what's enabled
- **Support quick disable** - flags must be toggleable without deployment
- **Clean up stale flags** - remove flags after features stabilize

### SHOULD

- **Support per-client overrides** - enable/disable per tenant or client
- **Log flag evaluations** - for debugging and analytics
- **Set expiration dates** - prevent flag accumulation
- **Test both flag states** - in CI/CD pipelines

### MAY

- **Support percentage rollouts** - gradual traffic increases
- **Integrate with external systems** - LaunchDarkly, Split, etc.
- **Support flag dependencies** - flag A requires flag B

## Feature Flag Lifecycle

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ EXPERIMENTAL │ ──▶ │     BETA     │ ──▶ │    STABLE    │ ──▶ │  DEPRECATED  │
│              │     │              │     │              │     │              │
│ - Opt-in     │     │ - Default    │     │ - Default    │     │ - Warn on    │
│ - May change │     │   off        │     │   on         │     │   use        │
│ - No SLA     │     │ - Opt-in     │     │ - Full SLA   │     │ - Remove     │
│              │     │ - Some SLA   │     │              │     │   after X    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Implementation

### Flag Definition

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Set, Dict, Any

class FlagState(str, Enum):
    EXPERIMENTAL = "experimental"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"

@dataclass
class FeatureFlag:
    name: str
    description: str
    state: FlagState
    default_enabled: bool
    created_at: datetime
    expires_at: Optional[datetime] = None
    owner: str = ""
    allowed_clients: Set[str] = field(default_factory=set)  # Empty = all
    blocked_clients: Set[str] = field(default_factory=set)
    percentage_rollout: float = 100.0  # 0-100
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Flag registry
FLAGS: Dict[str, FeatureFlag] = {
    "response_v3": FeatureFlag(
        name="response_v3",
        description="New response envelope format with enhanced metadata",
        state=FlagState.BETA,
        default_enabled=False,
        created_at=datetime(2025, 11, 1),
        expires_at=datetime(2026, 2, 1),
        owner="platform-team",
        percentage_rollout=25.0,
    ),
    "streaming_responses": FeatureFlag(
        name="streaming_responses",
        description="Enable streaming for large response payloads",
        state=FlagState.EXPERIMENTAL,
        default_enabled=False,
        created_at=datetime(2025, 11, 15),
        owner="performance-team",
        allowed_clients={"client_alpha", "client_beta"},
    ),
    "batch_operations": FeatureFlag(
        name="batch_operations",
        description="Support for batch tool invocations",
        state=FlagState.STABLE,
        default_enabled=True,
        created_at=datetime(2025, 6, 1),
        owner="api-team",
    ),
}
```

### Flag Evaluation

```python
import hashlib
import random
from typing import Optional
from contextvars import ContextVar

# Current client context
current_client_id: ContextVar[str] = ContextVar("client_id", default="anonymous")

class FeatureFlagService:
    """Evaluate feature flags with client context."""

    def __init__(self, flags: Dict[str, FeatureFlag]):
        self.flags = flags
        self._overrides: Dict[str, Dict[str, bool]] = {}  # client_id -> flag -> value

    def is_enabled(
        self,
        flag_name: str,
        client_id: Optional[str] = None,
        default: bool = False
    ) -> bool:
        """Check if feature flag is enabled for client."""
        client_id = client_id or current_client_id.get()

        # Check if flag exists
        flag = self.flags.get(flag_name)
        if not flag:
            logger.warning(f"Unknown feature flag: {flag_name}")
            return default

        # Check for client-specific override
        if client_id in self._overrides:
            if flag_name in self._overrides[client_id]:
                return self._overrides[client_id][flag_name]

        # Check if flag is deprecated (still works but warn)
        if flag.state == FlagState.DEPRECATED:
            logger.warning(
                f"Deprecated feature flag '{flag_name}' accessed",
                extra={"client_id": client_id, "flag": flag_name}
            )

        # Check expiration
        if flag.expires_at and datetime.utcnow() > flag.expires_at:
            logger.warning(f"Expired feature flag: {flag_name}")
            return default

        # Check client allowlist/blocklist
        if flag.blocked_clients and client_id in flag.blocked_clients:
            return False
        if flag.allowed_clients and client_id not in flag.allowed_clients:
            return False

        # Check dependencies
        for dep_flag in flag.dependencies:
            if not self.is_enabled(dep_flag, client_id):
                return False

        # Check percentage rollout (deterministic based on client_id)
        if flag.percentage_rollout < 100.0:
            hash_input = f"{flag_name}:{client_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            bucket = (hash_value % 100) + 1
            if bucket > flag.percentage_rollout:
                return False

        return flag.default_enabled

    def set_override(self, client_id: str, flag_name: str, enabled: bool):
        """Set client-specific flag override."""
        if client_id not in self._overrides:
            self._overrides[client_id] = {}
        self._overrides[client_id][flag_name] = enabled

    def clear_override(self, client_id: str, flag_name: str):
        """Clear client-specific flag override."""
        if client_id in self._overrides:
            self._overrides[client_id].pop(flag_name, None)

# Global service instance
flag_service = FeatureFlagService(FLAGS)
```

### Using Flags in Tools

```python
from functools import wraps

def feature_flag(flag_name: str, fallback=None):
    """Decorator to gate tool behind feature flag."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not flag_service.is_enabled(flag_name):
                if fallback:
                    return fallback(*args, **kwargs)
                return asdict(error_response(
                    error=f"Feature '{flag_name}' is not enabled",
                    data={
                        "error_code": "FEATURE_DISABLED",
                        "feature": flag_name,
                        "remediation": "Contact support to enable this feature"
                    }
                ))
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage: Entire tool behind flag
@mcp.tool()
@feature_flag("batch_operations")
def batch_process(items: List[dict]) -> dict:
    """Process items in batch (requires batch_operations flag)."""
    ...

# Usage: Conditional behavior within tool
@mcp.tool()
def get_user(user_id: str) -> dict:
    """Get user details."""
    user = db.get_user(user_id)

    response_data = {"user": user.to_dict()}

    # Add enhanced metadata if v3 response enabled
    if flag_service.is_enabled("response_v3"):
        response_data["_enhanced"] = {
            "computed_fields": user.computed_fields(),
            "relations": user.relation_summary()
        }

    return asdict(success_response(data=response_data))
```

## Exposing Flags in Capabilities

```python
@mcp.tool()
def get_capabilities(client_id: str = None) -> dict:
    """Get server capabilities including feature flags."""
    client_id = client_id or current_client_id.get()

    # Build flag status for this client
    flag_status = {}
    for name, flag in FLAGS.items():
        flag_status[name] = {
            "enabled": flag_service.is_enabled(name, client_id),
            "state": flag.state.value,
            "description": flag.description,
        }
        if flag.state == FlagState.DEPRECATED:
            flag_status[name]["deprecation_notice"] = (
                f"This feature is deprecated and will be removed after {flag.expires_at}"
            )

    return asdict(success_response(data={
        "capabilities": {
            "response_version": "response-v2",
            "streaming": flag_service.is_enabled("streaming_responses", client_id),
            "batch_operations": flag_service.is_enabled("batch_operations", client_id),
        },
        "feature_flags": flag_status,
        "server_version": "1.5.0"
    }))
```

## Gradual Rollout Patterns

### Percentage-Based Rollout

```python
# Start at 5%, increase gradually
flag = FLAGS["new_algorithm"]
flag.percentage_rollout = 5.0   # Day 1: 5%
flag.percentage_rollout = 25.0  # Day 3: 25%
flag.percentage_rollout = 50.0  # Day 7: 50%
flag.percentage_rollout = 100.0 # Day 14: 100%
```

### Allowlist Expansion

```python
# Start with specific clients, then open up
flag = FLAGS["streaming_responses"]
flag.allowed_clients = {"internal_testing"}  # Phase 1
flag.allowed_clients = {"internal_testing", "beta_users"}  # Phase 2
flag.allowed_clients = set()  # Phase 3: Open to all
flag.state = FlagState.STABLE
```

### Canary Deployment

```python
def is_canary_client(client_id: str) -> bool:
    """Check if client is in canary group."""
    # Use consistent hashing for stable canary assignment
    hash_value = int(hashlib.md5(client_id.encode()).hexdigest(), 16)
    return (hash_value % 100) < 5  # 5% canary

@mcp.tool()
def process_request(data: dict) -> dict:
    """Process with canary routing."""
    client_id = current_client_id.get()

    if is_canary_client(client_id) and flag_service.is_enabled("new_processor"):
        return new_processor(data)  # Canary path
    else:
        return legacy_processor(data)  # Stable path
```

## Flag Cleanup

```python
def audit_flags() -> dict:
    """Audit feature flags for cleanup opportunities."""
    now = datetime.utcnow()
    issues = []

    for name, flag in FLAGS.items():
        # Check for expired flags
        if flag.expires_at and now > flag.expires_at:
            issues.append({
                "flag": name,
                "issue": "expired",
                "expired_at": flag.expires_at.isoformat(),
                "action": "Remove flag and clean up code"
            })

        # Check for old stable flags (should be removed)
        age_days = (now - flag.created_at).days
        if flag.state == FlagState.STABLE and age_days > 90:
            issues.append({
                "flag": name,
                "issue": "stale_stable",
                "age_days": age_days,
                "action": "Flag is stable for 90+ days, consider removing"
            })

        # Check for long-running experimental flags
        if flag.state == FlagState.EXPERIMENTAL and age_days > 30:
            issues.append({
                "flag": name,
                "issue": "stale_experimental",
                "age_days": age_days,
                "action": "Promote to beta or remove"
            })

    return {
        "total_flags": len(FLAGS),
        "issues": issues,
        "flags_by_state": {
            state.value: len([f for f in FLAGS.values() if f.state == state])
            for state in FlagState
        }
    }
```

## Testing with Flags

```python
import pytest
from contextlib import contextmanager

@contextmanager
def flag_override(flag_name: str, enabled: bool):
    """Context manager for testing with specific flag state."""
    client_id = "test_client"
    flag_service.set_override(client_id, flag_name, enabled)
    try:
        yield
    finally:
        flag_service.clear_override(client_id, flag_name)

class TestFeatureFlags:
    """Test behavior with different flag states."""

    def test_feature_enabled(self):
        """Test behavior when flag is enabled."""
        with flag_override("batch_operations", True):
            result = batch_process([{"id": 1}, {"id": 2}])
            assert result["success"] is True

    def test_feature_disabled(self):
        """Test behavior when flag is disabled."""
        with flag_override("batch_operations", False):
            result = batch_process([{"id": 1}])
            assert result["success"] is False
            assert result["data"]["error_code"] == "FEATURE_DISABLED"

    @pytest.mark.parametrize("flag_enabled", [True, False])
    def test_both_flag_states(self, flag_enabled):
        """Test both flag states in CI."""
        with flag_override("new_algorithm", flag_enabled):
            result = process_data({"input": "test"})
            assert result["success"] is True  # Should work either way
```

## Anti-Patterns

### Don't: Use Flags for Permanent Conditionals

```python
# Bad: Flag that will never be removed
if flag_service.is_enabled("use_new_database"):  # Permanent architecture decision
    return new_db.query(...)
else:
    return old_db.query(...)

# Good: Make the decision, migrate, remove flag
return new_db.query(...)  # After migration complete
```

### Don't: Nest Flag Checks Deeply

```python
# Bad: Complex nested flags
if flag_service.is_enabled("feature_a"):
    if flag_service.is_enabled("feature_b"):
        if flag_service.is_enabled("feature_c"):
            ...

# Good: Combine into single flag or use dependencies
if flag_service.is_enabled("feature_abc_bundle"):
    ...
```

### Don't: Skip Flag Cleanup

```python
# Bad: Accumulating flags forever
FLAGS = {
    "feature_2023_q1": ...,  # 2 years old!
    "feature_2023_q2": ...,
    "feature_2024_q1": ...,
    # ... hundreds of flags
}

# Good: Regular cleanup with expiration dates
FLAGS = {
    "current_beta_feature": FeatureFlag(
        expires_at=datetime(2025, 3, 1),  # Must decide by this date
        ...
    )
}
```

## Related Documents

- [Versioned Contracts](./01-versioned-contracts.md) - Schema versioning with flags
- [Tool Discovery](./13-tool-discovery.md) - Exposing flags in capabilities
- [Testing & Fixtures](./10-testing-fixtures.md) - Testing flag states

---

**Navigation:** [← Tool Discovery](./13-tool-discovery.md) | [Index](./README.md) | [Next: Concurrency Patterns →](./15-concurrency-patterns.md)
