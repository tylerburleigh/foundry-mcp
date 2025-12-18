"""Metrics page - viewer with summaries."""

import streamlit as st

from foundry_mcp.dashboard.components.filters import time_range_filter
from foundry_mcp.dashboard.components.cards import kpi_row
from foundry_mcp.dashboard.data.stores import (
    get_metrics_list,
    get_metrics_timeseries,
    get_metrics_summary,
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
    """Render the Metrics page."""
    st.header("Metrics")

    # Get available metrics
    metrics_list = get_metrics_list()

    if not metrics_list:
        st.warning("No metrics available. Metrics persistence may be disabled.")
        st.info("Enable metrics persistence in foundry-mcp.toml under [metrics_persistence]")
        return

    # Metric selector and time range
    col1, col2 = st.columns([2, 1])

    with col1:
        metric_names = [m.get("metric_name", "unknown") for m in metrics_list]
        selected_metric = st.selectbox(
            "Select Metric",
            options=metric_names,
            key="metrics_selector",
        )

    with col2:
        hours = time_range_filter(key="metrics_time_range", default="24h")

    st.divider()

    if selected_metric:
        # Get summary statistics
        summary = get_metrics_summary(selected_metric, since_hours=hours)

        # Summary cards
        st.subheader("Summary Statistics")
        if summary.get("enabled"):
            # Handle None values (returned when no data exists)
            min_val = summary.get("min") if summary.get("min") is not None else 0
            max_val = summary.get("max") if summary.get("max") is not None else 0
            avg_val = summary.get("avg") if summary.get("avg") is not None else 0
            sum_val = summary.get("sum") if summary.get("sum") is not None else 0

            kpi_row(
                [
                    {"label": "Count", "value": summary.get("count", 0)},
                    {"label": "Min", "value": f"{min_val:.2f}"},
                    {"label": "Max", "value": f"{max_val:.2f}"},
                    {"label": "Average", "value": f"{avg_val:.2f}"},
                    {"label": "Sum", "value": f"{sum_val:.2f}"},
                ],
                columns=5,
            )
        else:
            st.info("Summary not available")

        st.divider()

        # Data table
        st.subheader(f"Data: {selected_metric}")
        timeseries_df = get_metrics_timeseries(selected_metric, since_hours=hours)

        # Check if we have data, if not try longer time ranges
        display_df = timeseries_df
        time_range_note = None

        if display_df is None or display_df.empty:
            # Try progressively longer time ranges
            for fallback_hours, label in [(168, "7 days"), (720, "30 days"), (8760, "1 year")]:
                if fallback_hours > hours:
                    display_df = get_metrics_timeseries(selected_metric, since_hours=fallback_hours)
                    if display_df is not None and not display_df.empty:
                        time_range_note = f"No data in selected range - showing last {label}"
                        break

        if display_df is not None and not display_df.empty:
            if time_range_note:
                st.caption(time_range_note)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            # Export
            col1, col2 = st.columns(2)
            with col1:
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_metric}_export.csv",
                    mime="text/csv",
                )
            with col2:
                json_data = display_df.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{selected_metric}_export.json",
                    mime="application/json",
                )
        else:
            st.info(f"No data available for {selected_metric}")

    # Tool Action Breakdown
    st.divider()
    st.subheader("Top Tool Actions")
    top_actions = get_top_tool_actions(since_hours=hours, top_n=10)

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
        st.info("No tool action data available for selected time range")

    # Metrics catalog
    st.divider()
    st.subheader("Available Metrics")

    if PANDAS_AVAILABLE:
        metrics_df = pd.DataFrame(metrics_list)
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "metric_name": st.column_config.TextColumn("Metric", width="medium"),
                "count": st.column_config.NumberColumn("Records", width="small"),
            },
        )
    else:
        for m in metrics_list:
            st.text(f"- {m.get('metric_name', 'unknown')} ({m.get('count', 0)} records)")
