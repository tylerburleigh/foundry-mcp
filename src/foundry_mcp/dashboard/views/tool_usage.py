"""Tool Usage page - detailed breakdown of tool and action invocations."""

import streamlit as st

from foundry_mcp.dashboard.components.filters import time_range_filter
from foundry_mcp.dashboard.components.charts import bar_chart, pie_chart, empty_chart
from foundry_mcp.dashboard.components.cards import kpi_row
from foundry_mcp.dashboard.data.stores import (
    get_tool_action_breakdown,
    get_top_tool_actions,
)

# Try importing pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


def render():
    """Render the Tool Usage page."""
    st.header("Tool Usage")

    # Time range filter
    col1, col2 = st.columns([3, 1])
    with col2:
        hours = time_range_filter(key="tool_usage_time_range", default="24h")

    st.divider()

    # Get breakdown data
    breakdown_df = get_tool_action_breakdown(since_hours=hours)

    if breakdown_df is None or breakdown_df.empty:
        st.warning("No tool usage data available. Data will appear once tools are invoked.")
        st.info("Tool invocations are recorded when MCP tools are called.")
        return

    # KPI Summary
    total_invocations = int(breakdown_df["count"].sum())
    unique_tools = breakdown_df["tool"].nunique()
    unique_actions = breakdown_df[breakdown_df["action"] != "(no action)"]["action"].nunique()
    success_count = int(breakdown_df[breakdown_df["status"] == "success"]["count"].sum())
    success_rate = (success_count / total_invocations * 100) if total_invocations > 0 else 0

    kpi_row(
        [
            {"label": "Total Invocations", "value": f"{total_invocations:,}"},
            {"label": "Unique Tools", "value": unique_tools},
            {"label": "Unique Actions", "value": unique_actions},
            {"label": "Success Rate", "value": f"{success_rate:.1f}%"},
        ],
        columns=4,
    )

    st.divider()

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Tools")
        tool_totals = breakdown_df.groupby("tool")["count"].sum().reset_index()
        tool_totals = tool_totals.sort_values("count", ascending=True).tail(10)
        if not tool_totals.empty:
            bar_chart(tool_totals, x="count", y="tool", orientation="h", height=350)
        else:
            empty_chart("No tool data")

    with col2:
        st.subheader("Status Distribution")
        status_totals = breakdown_df.groupby("status")["count"].sum().reset_index()
        if not status_totals.empty:
            pie_chart(status_totals, values="count", names="status", height=350)
        else:
            empty_chart("No status data")

    st.divider()

    # Detailed Action Breakdown
    st.subheader("Action Breakdown by Tool")

    # Tool selector
    tools = sorted(breakdown_df["tool"].unique())
    selected_tool = st.selectbox(
        "Select Tool",
        options=["All Tools"] + tools,
        key="tool_usage_tool_selector",
    )

    if selected_tool == "All Tools":
        filtered_df = breakdown_df
    else:
        filtered_df = breakdown_df[breakdown_df["tool"] == selected_tool]

    if not filtered_df.empty and PANDAS_AVAILABLE:
        # Create display name column
        action_df = filtered_df.groupby(["tool", "action", "status"])["count"].sum().reset_index()
        action_df["display_name"] = action_df.apply(
            lambda r: f"{r['tool']}.{r['action']}" if r["action"] != "(no action)" else r["tool"],
            axis=1
        )

        # Detailed table
        st.subheader("Detailed Data")
        display_df = action_df[["display_name", "status", "count"]].sort_values("count", ascending=False)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "display_name": st.column_config.TextColumn("Tool.Action", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "count": st.column_config.NumberColumn("Invocations", width="small"),
            },
        )

        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="tool_usage_export.csv",
                mime="text/csv",
            )
        with col2:
            json_data = display_df.to_json(orient="records")
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="tool_usage_export.json",
                mime="application/json",
            )
    else:
        empty_chart("No data for selected filter")
