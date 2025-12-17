"""Overview page - dashboard home with KPIs and summary charts."""

import streamlit as st

from foundry_mcp.dashboard.components.cards import kpi_row
from foundry_mcp.dashboard.components.charts import line_chart, empty_chart
from foundry_mcp.dashboard.data.stores import (
    get_overview_summary,
    get_metrics_timeseries,
    get_errors,
    get_error_patterns,
    get_top_tool_actions,
)


def render():
    """Render the Overview page."""
    st.header("Overview")

    # Get summary data
    summary = get_overview_summary()

    # KPI Cards Row
    st.subheader("Key Metrics")
    kpi_row(
        [
            {
                "label": "Total Invocations",
                "value": summary.get("total_invocations", 0),
                "help": "Total tool invocations recorded (all time)",
            },
            {
                "label": "Total Errors",
                "value": summary.get("error_count", 0),
                "help": "Total errors recorded (all time)",
            },
        ],
        columns=2,
    )

    st.divider()

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tool Invocations (Last 24h)")
        invocations_df = get_metrics_timeseries("tool_invocations_total", since_hours=24)
        if invocations_df is not None and not invocations_df.empty:
            line_chart(
                invocations_df,
                x="timestamp",
                y="value",
                title=None,
                height=300,
            )
        else:
            # Try with longer time range if 24h is empty
            invocations_df_7d = get_metrics_timeseries("tool_invocations_total", since_hours=168)
            if invocations_df_7d is not None and not invocations_df_7d.empty:
                st.caption("No data in last 24h - showing last 7 days")
                line_chart(
                    invocations_df_7d,
                    x="timestamp",
                    y="value",
                    title=None,
                    height=300,
                )
            else:
                empty_chart("No invocation data available (metrics may be older than 7 days)")

    with col2:
        st.subheader("Error Rate (Last 24h)")
        errors_df = get_errors(since_hours=24, limit=500)

        # Try fallback to 7 days if 24h is empty
        time_note = None
        if errors_df is None or errors_df.empty:
            errors_df = get_errors(since_hours=168, limit=500)
            if errors_df is not None and not errors_df.empty:
                time_note = "No data in last 24h - showing last 7 days"

        if errors_df is not None and not errors_df.empty:
            if time_note:
                st.caption(time_note)
            # Group by hour for error rate
            try:
                errors_df["hour"] = errors_df["timestamp"].dt.floor("H")
                hourly_errors = errors_df.groupby("hour").size().reset_index(name="count")
                hourly_errors.columns = ["timestamp", "value"]
                line_chart(
                    hourly_errors,
                    x="timestamp",
                    y="value",
                    title=None,
                    height=300,
                )
            except Exception:
                empty_chart("Could not process error data")
        else:
            empty_chart("No error data available")

    st.divider()

    # Bottom Row - Patterns and Recent
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Error Patterns (All Time)")
        patterns = get_error_patterns(min_count=2)
        if patterns:
            for i, p in enumerate(patterns[:5]):
                with st.container(border=True):
                    st.markdown(f"**{p.get('tool_name', 'Unknown')}**")
                    st.caption(f"Count: {p.get('count', 0)} | Code: {p.get('error_code', 'N/A')}")
                    if p.get("message"):
                        st.text(p["message"][:100] + "..." if len(p.get("message", "")) > 100 else p.get("message", ""))
        else:
            st.info("No recurring error patterns detected")

    with col2:
        st.subheader("Recent Errors (Last Hour)")
        errors_df = get_errors(since_hours=1, limit=5)

        # Try fallback to 24h if 1h is empty
        if errors_df is None or errors_df.empty:
            errors_df = get_errors(since_hours=24, limit=5)
            if errors_df is not None and not errors_df.empty:
                st.caption("No errors in last hour - showing last 24h")

        if errors_df is not None and not errors_df.empty:
            for _, row in errors_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row.get('tool_name', 'Unknown')}**")
                    st.caption(f"{row.get('timestamp', '')} | {row.get('error_code', 'N/A')}")
                    msg = row.get("message", "")
                    st.text(msg[:80] + "..." if len(msg) > 80 else msg)
        else:
            st.info("No recent errors")

    st.divider()

    # Tool Usage Breakdown
    st.subheader("Top Tool Actions (Last 24h)")
    top_actions = get_top_tool_actions(since_hours=24, top_n=10)

    if top_actions:
        col1, col2 = st.columns(2)
        for i, item in enumerate(top_actions):
            tool = item.get("tool", "unknown")
            action = item.get("action", "")
            count = int(item.get("count", 0))
            display_name = f"{tool}.{action}" if action and action != "(no action)" else tool
            target_col = col1 if i < 5 else col2
            with target_col:
                with st.container(border=True):
                    st.markdown(f"**{display_name}**")
                    st.caption(f"Invocations: {count:,}")
    else:
        st.info("No tool action data available yet")
