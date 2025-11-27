"""Tests for feature flags infrastructure.

Tests cover flag registration, lookup, evaluation logic, percentage rollouts,
and testing utilities (flag_override context manager).
"""

from datetime import datetime, timedelta, timezone

import pytest

from foundry_mcp.core.feature_flags import (
    FeatureFlag,
    FeatureFlagRegistry,
    FlagState,
    current_client_id,
    feature_flag,
    flag_override,
    get_registry,
)


class TestFlagRegistrationAndLookup:
    """Tests for flag registration and lookup operations."""

    def test_register_flag(self) -> None:
        """Test basic flag registration."""
        registry = FeatureFlagRegistry()
        flag = FeatureFlag(
            name="test_flag",
            description="Test flag",
            state=FlagState.BETA,
            default_enabled=False,
        )

        registry.register(flag)
        retrieved = registry.get("test_flag")

        assert retrieved is not None
        assert retrieved.name == "test_flag"
        assert retrieved.description == "Test flag"
        assert retrieved.state == FlagState.BETA
        assert retrieved.default_enabled is False

    def test_register_duplicate_flag_raises(self) -> None:
        """Test that registering a duplicate flag raises ValueError."""
        registry = FeatureFlagRegistry()
        flag = FeatureFlag(
            name="test_flag",
            description="Test flag",
            state=FlagState.BETA,
            default_enabled=False,
        )

        registry.register(flag)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(flag)

    def test_get_nonexistent_flag(self) -> None:
        """Test that getting a nonexistent flag returns None."""
        registry = FeatureFlagRegistry()
        assert registry.get("nonexistent") is None

    def test_list_flags(self) -> None:
        """Test listing all registered flags."""
        registry = FeatureFlagRegistry()
        flag1 = FeatureFlag(
            name="flag_a",
            description="Flag A",
            state=FlagState.STABLE,
            default_enabled=True,
        )
        flag2 = FeatureFlag(
            name="flag_b",
            description="Flag B",
            state=FlagState.EXPERIMENTAL,
            default_enabled=False,
        )

        registry.register(flag1)
        registry.register(flag2)

        all_flags = registry.list_flags()
        assert len(all_flags) == 2
        assert "flag_a" in all_flags
        assert "flag_b" in all_flags

    def test_is_enabled_unknown_flag(self) -> None:
        """Test that checking unknown flag returns default."""
        registry = FeatureFlagRegistry()

        # Should return False (default default)
        assert registry.is_enabled("unknown") is False

        # Should return custom default
        assert registry.is_enabled("unknown", default=True) is True

    def test_is_enabled_basic(self) -> None:
        """Test basic enabled/disabled flags."""
        registry = FeatureFlagRegistry()

        enabled_flag = FeatureFlag(
            name="enabled_flag",
            description="Enabled flag",
            state=FlagState.STABLE,
            default_enabled=True,
        )
        disabled_flag = FeatureFlag(
            name="disabled_flag",
            description="Disabled flag",
            state=FlagState.BETA,
            default_enabled=False,
        )

        registry.register(enabled_flag)
        registry.register(disabled_flag)

        assert registry.is_enabled("enabled_flag") is True
        assert registry.is_enabled("disabled_flag") is False

    def test_flag_state_enum_values(self) -> None:
        """Test FlagState enum has all expected values."""
        assert FlagState.EXPERIMENTAL.value == "experimental"
        assert FlagState.BETA.value == "beta"
        assert FlagState.STABLE.value == "stable"
        assert FlagState.DEPRECATED.value == "deprecated"

    def test_feature_flag_to_dict(self) -> None:
        """Test FeatureFlag serialization to dict."""
        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        expires = datetime(2025, 12, 31, tzinfo=timezone.utc)

        flag = FeatureFlag(
            name="my_flag",
            description="My flag description",
            state=FlagState.BETA,
            default_enabled=False,
            created_at=created,
            expires_at=expires,
            owner="test-team",
            percentage_rollout=50.0,
            allowed_clients={"client_a", "client_b"},
            blocked_clients={"client_x"},
            dependencies={"other_flag"},
            metadata={"key": "value"},
        )

        data = flag.to_dict()

        assert data["name"] == "my_flag"
        assert data["description"] == "My flag description"
        assert data["state"] == "beta"
        assert data["default_enabled"] is False
        assert data["created_at"] == "2025-01-01T00:00:00+00:00"
        assert data["expires_at"] == "2025-12-31T00:00:00+00:00"
        assert data["owner"] == "test-team"
        assert data["percentage_rollout"] == 50.0
        assert set(data["allowed_clients"]) == {"client_a", "client_b"}
        assert data["blocked_clients"] == ["client_x"]
        assert data["dependencies"] == ["other_flag"]
        assert data["metadata"] == {"key": "value"}

    def test_is_expired(self) -> None:
        """Test flag expiration detection."""
        now = datetime.now(timezone.utc)

        # Not expired (no expiration date)
        flag_no_expiry = FeatureFlag(
            name="no_expiry",
            description="No expiry",
            state=FlagState.STABLE,
            default_enabled=True,
        )
        assert flag_no_expiry.is_expired() is False

        # Not expired (future date)
        flag_future = FeatureFlag(
            name="future",
            description="Future",
            state=FlagState.STABLE,
            default_enabled=True,
            expires_at=now + timedelta(days=30),
        )
        assert flag_future.is_expired() is False

        # Expired (past date)
        flag_expired = FeatureFlag(
            name="expired",
            description="Expired",
            state=FlagState.DEPRECATED,
            default_enabled=True,
            expires_at=now - timedelta(days=1),
        )
        assert flag_expired.is_expired() is True

    def test_expired_flag_returns_default(self) -> None:
        """Test that expired flags return the default value."""
        registry = FeatureFlagRegistry()
        now = datetime.now(timezone.utc)

        expired_flag = FeatureFlag(
            name="expired_flag",
            description="Expired flag",
            state=FlagState.DEPRECATED,
            default_enabled=True,
            expires_at=now - timedelta(days=1),
        )
        registry.register(expired_flag)

        # Expired flag should return False (not the flag's default_enabled)
        assert registry.is_enabled("expired_flag") is False
        assert registry.is_enabled("expired_flag", default=True) is True

    def test_client_override_takes_precedence(self) -> None:
        """Test that client overrides take precedence over flag settings."""
        registry = FeatureFlagRegistry()
        flag = FeatureFlag(
            name="test_flag",
            description="Test",
            state=FlagState.STABLE,
            default_enabled=False,
        )
        registry.register(flag)

        assert registry.is_enabled("test_flag", client_id="client_a") is False

        # Set override
        registry.set_override("client_a", "test_flag", True)
        assert registry.is_enabled("test_flag", client_id="client_a") is True

        # Other clients unaffected
        assert registry.is_enabled("test_flag", client_id="client_b") is False

        # Clear override
        registry.clear_override("client_a", "test_flag")
        assert registry.is_enabled("test_flag", client_id="client_a") is False

    def test_blocklist_denies_access(self) -> None:
        """Test that blocked clients cannot access flag."""
        registry = FeatureFlagRegistry()
        flag = FeatureFlag(
            name="test_flag",
            description="Test",
            state=FlagState.STABLE,
            default_enabled=True,
            blocked_clients={"blocked_client"},
        )
        registry.register(flag)

        assert registry.is_enabled("test_flag", client_id="regular_client") is True
        assert registry.is_enabled("test_flag", client_id="blocked_client") is False

    def test_allowlist_restricts_access(self) -> None:
        """Test that allowlist restricts access to listed clients only."""
        registry = FeatureFlagRegistry()
        flag = FeatureFlag(
            name="test_flag",
            description="Test",
            state=FlagState.STABLE,
            default_enabled=True,
            allowed_clients={"allowed_client"},
        )
        registry.register(flag)

        assert registry.is_enabled("test_flag", client_id="allowed_client") is True
        assert registry.is_enabled("test_flag", client_id="other_client") is False

    def test_flag_dependencies(self) -> None:
        """Test that flag dependencies are evaluated."""
        registry = FeatureFlagRegistry()

        # Create base flag that's enabled
        base_flag = FeatureFlag(
            name="base_flag",
            description="Base",
            state=FlagState.STABLE,
            default_enabled=True,
        )

        # Create dependent flag
        dependent_flag = FeatureFlag(
            name="dependent_flag",
            description="Dependent",
            state=FlagState.STABLE,
            default_enabled=True,
            dependencies={"base_flag"},
        )

        # Create disabled base
        disabled_base = FeatureFlag(
            name="disabled_base",
            description="Disabled base",
            state=FlagState.STABLE,
            default_enabled=False,
        )

        # Create flag depending on disabled base
        blocked_by_dep = FeatureFlag(
            name="blocked_flag",
            description="Blocked by dependency",
            state=FlagState.STABLE,
            default_enabled=True,
            dependencies={"disabled_base"},
        )

        registry.register(base_flag)
        registry.register(dependent_flag)
        registry.register(disabled_base)
        registry.register(blocked_by_dep)

        # dependent_flag should work since base_flag is enabled
        assert registry.is_enabled("dependent_flag") is True

        # blocked_flag should be disabled due to disabled dependency
        assert registry.is_enabled("blocked_flag") is False

    def test_get_flags_for_capabilities(self) -> None:
        """Test getting flag status for capabilities endpoint."""
        registry = FeatureFlagRegistry()

        stable_flag = FeatureFlag(
            name="stable_feature",
            description="A stable feature",
            state=FlagState.STABLE,
            default_enabled=True,
        )

        deprecated_flag = FeatureFlag(
            name="deprecated_feature",
            description="A deprecated feature",
            state=FlagState.DEPRECATED,
            default_enabled=True,
            expires_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        registry.register(stable_flag)
        registry.register(deprecated_flag)

        capabilities = registry.get_flags_for_capabilities(client_id="test_client")

        assert "stable_feature" in capabilities
        assert capabilities["stable_feature"]["enabled"] is True
        assert capabilities["stable_feature"]["state"] == "stable"
        assert capabilities["stable_feature"]["description"] == "A stable feature"

        assert "deprecated_feature" in capabilities
        assert capabilities["deprecated_feature"]["state"] == "deprecated"
        assert "deprecation_notice" in capabilities["deprecated_feature"]

    def test_global_registry(self) -> None:
        """Test the global registry singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_clear_all_overrides(self) -> None:
        """Test clearing all overrides."""
        registry = FeatureFlagRegistry()
        flag = FeatureFlag(
            name="test_flag",
            description="Test",
            state=FlagState.STABLE,
            default_enabled=False,
        )
        registry.register(flag)

        registry.set_override("client_a", "test_flag", True)
        registry.set_override("client_b", "test_flag", True)

        assert registry.is_enabled("test_flag", client_id="client_a") is True
        assert registry.is_enabled("test_flag", client_id="client_b") is True

        # Clear only client_a
        registry.clear_all_overrides(client_id="client_a")
        assert registry.is_enabled("test_flag", client_id="client_a") is False
        assert registry.is_enabled("test_flag", client_id="client_b") is True

        # Clear all
        registry.clear_all_overrides()
        assert registry.is_enabled("test_flag", client_id="client_b") is False
