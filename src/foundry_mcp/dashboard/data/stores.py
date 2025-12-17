"""Data access layer for dashboard.

Wraps MetricsStore, ErrorStore, and other data sources with:
- Singleton store instances (stores handle internal caching)
- pandas DataFrame conversion for easy use with st.dataframe
- Graceful handling when stores are disabled
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Try importing pandas - it's an optional dependency
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def _get_config():
    """Get foundry-mcp configuration."""
    try:
        from foundry_mcp.config import get_config

        return get_config()
    except Exception as e:
        logger.warning("Could not load config: %s", e)
        return None


def _get_error_store():
    """Get error store singleton instance if enabled."""
    config = _get_config()
    if config is None or not config.error_collection.enabled:
        return None

    try:
        from foundry_mcp.core.error_store import get_error_store

        storage_path = config.error_collection.get_storage_path()
        return get_error_store(storage_path)  # Returns singleton
    except Exception as e:
        logger.warning("Could not initialize error store: %s", e)
        return None


def _get_metrics_store():
    """Get metrics store singleton instance if enabled."""
    config = _get_config()
    if config is None or not config.metrics_persistence.enabled:
        return None

    try:
        from foundry_mcp.core.metrics_store import get_metrics_store

        storage_path = config.metrics_persistence.get_storage_path()
        return get_metrics_store(storage_path)  # Returns singleton
    except Exception as e:
        logger.warning("Could not initialize metrics store: %s", e)
        return None


# =============================================================================
# Error Data Functions
# =============================================================================


def get_errors(
    tool_name: Optional[str] = None,
    error_code: Optional[str] = None,
    since_hours: int = 24,
    limit: int = 100,
) -> "pd.DataFrame":
    """Get errors as a DataFrame.

    Args:
        tool_name: Filter by tool name
        error_code: Filter by error code
        since_hours: Hours to look back
        limit: Maximum records to return

    Returns:
        DataFrame with error records, or empty DataFrame if disabled
    """
    if not PANDAS_AVAILABLE:
        return None

    store = _get_error_store()
    if store is None:
        return pd.DataFrame()

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()

    try:
        records = store.query(
            tool_name=tool_name,
            error_code=error_code,
            since=since,
            limit=limit,
        )

        if not records:
            return pd.DataFrame()

        # Convert to list of dicts
        data = []
        for r in records:
            data.append(
                {
                    "id": r.error_id,
                    "timestamp": r.timestamp,
                    "tool_name": r.tool_name,
                    "error_code": r.error_code,
                    "message": r.message,
                    "error_type": r.error_type,
                    "fingerprint": r.fingerprint,
                }
            )

        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except Exception as e:
        logger.exception("Error querying errors: %s", e)
        return pd.DataFrame()


def get_error_stats() -> dict[str, Any]:
    """Get error statistics."""
    store = _get_error_store()
    if store is None:
        return {"enabled": False, "total": 0}

    try:
        stats = store.get_stats()
        stats["enabled"] = True
        return stats
    except Exception as e:
        logger.exception("Error getting error stats: %s", e)
        return {"enabled": True, "error": str(e)}


def get_error_patterns(min_count: int = 3) -> list[dict]:
    """Get recurring error patterns."""
    store = _get_error_store()
    if store is None:
        return []

    try:
        return store.get_patterns(min_count=min_count)
    except Exception as e:
        logger.exception("Error getting patterns: %s", e)
        return []


def get_error_by_id(error_id: str) -> Optional[dict]:
    """Get a single error by ID (not cached for freshness)."""
    store = _get_error_store()
    if store is None:
        return None

    try:
        record = store.get(error_id)
        if record is None:
            return None

        return {
            "id": record.error_id,
            "timestamp": record.timestamp,
            "tool_name": record.tool_name,
            "error_code": record.error_code,
            "message": record.message,
            "error_type": record.error_type,
            "fingerprint": record.fingerprint,
            "stack_trace": record.stack_trace,
            "context": record.context,
        }
    except Exception as e:
        logger.exception("Error getting error by ID: %s", e)
        return None


# =============================================================================
# Metrics Data Functions
# =============================================================================


def get_metrics_list() -> list[dict]:
    """Get list of available metrics."""
    store = _get_metrics_store()
    if store is None:
        return []

    try:
        return store.list_metrics()
    except Exception as e:
        logger.exception("Error listing metrics: %s", e)
        return []


def get_metrics_timeseries(
    metric_name: str,
    since_hours: int = 24,
    limit: int = 1000,
) -> "pd.DataFrame":
    """Get time-series data for a metric.

    Args:
        metric_name: Name of the metric
        since_hours: Hours to look back
        limit: Maximum data points

    Returns:
        DataFrame with timestamp and value columns
    """
    if not PANDAS_AVAILABLE:
        return None

    store = _get_metrics_store()
    if store is None:
        return pd.DataFrame()

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()

    try:
        data_points = store.query(
            metric_name=metric_name,
            since=since,
            limit=limit,
        )

        if not data_points:
            return pd.DataFrame()

        data = [
            {
                "timestamp": dp.timestamp,
                "value": dp.value,
                "labels": str(dp.labels) if dp.labels else "",
            }
            for dp in data_points
        ]

        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except Exception as e:
        logger.exception("Error querying metrics: %s", e)
        return pd.DataFrame()


def get_metrics_summary(metric_name: str, since_hours: int = 24) -> dict[str, Any]:
    """Get summary statistics for a metric."""
    store = _get_metrics_store()
    if store is None:
        return {"enabled": False}

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()

    try:
        summary = store.get_summary(metric_name, since=since)
        summary["enabled"] = True
        return summary
    except Exception as e:
        logger.exception("Error getting summary: %s", e)
        return {"enabled": True, "error": str(e)}


# =============================================================================
# Tool Usage Breakdown Functions
# =============================================================================


def get_tool_action_breakdown(
    since_hours: int = 24,
    limit: int = 1000,
) -> "pd.DataFrame":
    """Get tool invocations broken down by tool and action.

    Args:
        since_hours: Hours to look back
        limit: Maximum data points to query

    Returns:
        DataFrame with tool, action, status, and count columns
    """
    if not PANDAS_AVAILABLE:
        return None

    store = _get_metrics_store()
    if store is None:
        return pd.DataFrame()

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()

    try:
        data_points = store.query(
            metric_name="tool_invocations_total",
            since=since,
            limit=limit,
        )

        if not data_points:
            return pd.DataFrame()

        # Group by tool, action, status
        breakdown: dict[tuple[str, str, str], float] = {}
        for dp in data_points:
            tool = dp.labels.get("tool", "unknown")
            action = dp.labels.get("action", "")  # Empty string for legacy data
            status = dp.labels.get("status", "unknown")
            key = (tool, action, status)
            breakdown[key] = breakdown.get(key, 0) + dp.value

        data = [
            {"tool": k[0], "action": k[1] or "(no action)", "status": k[2], "count": int(v)}
            for k, v in breakdown.items()
        ]

        return pd.DataFrame(data)

    except Exception as e:
        logger.exception("Error getting tool action breakdown: %s", e)
        return pd.DataFrame()


def get_top_tool_actions(since_hours: int = 24, top_n: int = 10) -> list[dict]:
    """Get top N most called tool+action combinations.

    Args:
        since_hours: Hours to look back
        top_n: Number of top items to return

    Returns:
        List of dicts with tool, action, count
    """
    df = get_tool_action_breakdown(since_hours=since_hours)
    if df is None or df.empty:
        return []

    try:
        # Group by tool+action, sum counts across statuses
        grouped = df.groupby(["tool", "action"])["count"].sum().reset_index()
        grouped = grouped.sort_values("count", ascending=False).head(top_n)

        return grouped.to_dict("records")
    except Exception as e:
        logger.exception("Error getting top tool actions: %s", e)
        return []


# =============================================================================
# Health & Provider Data Functions
# =============================================================================


@st.cache_data(ttl=10)
def get_health_status() -> dict[str, Any]:
    """Get server health status."""
    try:
        from foundry_mcp.core.health import get_health_manager

        manager = get_health_manager()
        result = manager.check_health()

        return {
            "healthy": result.healthy,
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "checks": {
                name: {
                    "healthy": check.healthy,
                    "message": check.message,
                }
                for name, check in (result.checks or {}).items()
            },
        }
    except Exception as e:
        logger.exception("Error getting health: %s", e)
        return {"healthy": False, "error": str(e)}


@st.cache_data(ttl=30)
def get_providers() -> list[dict]:
    """Get list of AI providers with status."""
    try:
        from foundry_mcp.core.providers import describe_providers

        return describe_providers()
    except ImportError:
        # Providers module may not exist yet
        return []
    except Exception as e:
        logger.exception("Error getting providers: %s", e)
        return []


# =============================================================================
# Overview Summary Functions
# =============================================================================


def get_overview_summary() -> dict[str, Any]:
    """Get aggregated overview metrics for dashboard."""
    summary = {
        "total_invocations": 0,
        "error_count": 0,
    }

    # Get metrics summary
    metrics_list = get_metrics_list()
    for m in metrics_list:
        if m.get("metric_name") == "tool_invocations_total":
            summary["total_invocations"] = m.get("count", 0)

    # Get error count from store (single source of truth)
    error_store = _get_error_store()
    if error_store is not None:
        summary["error_count"] = error_store.get_total_count()

    return summary
