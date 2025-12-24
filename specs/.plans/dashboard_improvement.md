# Dashboard Overhaul Plan

## Context
The Streamlit dashboard at http://127.0.0.1:8501/ shows stale data, feels sluggish, and the UI hierarchy/presentation is not ideal. The goal is to refresh data reliably, improve responsiveness, and redesign layout/aggregation for better utility.

## Goals
- Keep data fresh and visibly up-to-date (10s cadence), without blocking the UI.
- Improve perceived and actual performance with caching and smarter queries.
- Redesign layout and information hierarchy for clarity and faster insight.
- Preserve current Streamlit-based architecture (no new UI layer).

## Non-Goals
- Replace Streamlit with a new frontend (React/Next/etc.).
- Rebuild the metrics/error persistence backends from scratch.
- Add multi-user auth or remote access.

## Current Issues (Observed)
- Data staleness: dashboard process does not consistently see new metrics/errors.
- Sluggish UX: blocking refresh loop and heavy re-reads on every interaction.
- "Vanilla" UI: weak hierarchy, inconsistent aggregation choices, limited context.

## Root Causes (Likely)
- In-memory indexes in `FileMetricsStore` / `FileErrorStore` are built once and not refreshed when files update (stale read risk across processes).
- Auto-refresh uses `time.sleep(...)` in the main Streamlit script, blocking interactivity.
- Data layer lacks consistent caching and query scoping, causing full scans per widget change.

## Approach Summary
- Freshness: detect data file changes and refresh indexes or data cache when updated.
- Performance: use `st.cache_data` with short TTL for heavy queries; provide cache bust.
- UX: establish a clear hierarchy (global controls, headline KPIs, focused panels).
- Autorefresh: add a native 10s refresh mechanism without blocking the UI.

## Planned Workstreams

### 1) Data Freshness + Caching
- Add file mtime tracking for metrics/errors JSONL in data layer.
- On mtime change, rebuild or refresh store index data used by dashboard queries.
- Add `st.cache_data(ttl=10)` for:
  - `get_errors`, `get_error_patterns`, `get_error_stats`
  - `get_metrics_list`, `get_metrics_timeseries`, `get_metrics_summary`
  - `get_tool_action_breakdown`, `get_top_tool_actions`
- Add a "Refresh Now" action that clears cached data and reruns.
- Surface last updated timestamps on key pages.

### 2) Autorefresh (10s)
- Replace blocking `time.sleep` with a Streamlit-safe auto-refresh:
  - Use `st.components.v1.html` to trigger a browser refresh every 10s when enabled.
  - Provide a toggle in the sidebar plus status indicator.

### 3) UX/UI Restructure
- Introduce a global control bar for time range and tool/status filters.
- Overview:
  - Top row KPIs (last 1h / 24h), success rate, latency p50/p95.
  - Trend strips for invocations and errors.
  - Recent errors and top tool actions as secondary panels.
- Errors:
  - Replace manual ID copy/paste with clickable row details.
  - Add quick filters (tool, error_code) and a summary bar.
- Tool Usage:
  - Surface tool/action success rate and latency summary.
  - Provide clearer aggregation with per-tool cards and action drilldown.
- Metrics:
  - Default to a curated list of core metrics; hide long raw lists behind an expander.
- Providers + SDD:
  - Wire into navigation and make them consistent with global filters.

### 4) Robustness & Diagnostics
- Add a "Data Status" panel: config flags, store enabled states, storage paths.
- Log dashboard refresh events (structured logs) for debugging.
- Improve launcher behavior if dashboard fails to start (surface stderr in CLI).

## Deliverables
- Updated dashboard app, data layer, and views.
- A refreshed UI layout with consistent hierarchy and styling.
- Documentation updates if needed (dashboard usage notes).

## Acceptance Criteria
- Data updates appear within 10 seconds of new writes (with refresh enabled).
- No UI blocking during refresh; user interactions remain responsive.
- Clear visual hierarchy: KPIs first, trends second, details last.
- Errors and tool usage are actionable without manual ID copy/paste.
- Dashboard continues to run with optional deps missing (graceful messages).

## Risks / Mitigations
- Risk: frequent file scans degrade performance.
  - Mitigation: short TTL caching and mtime checks before rebuilds.
- Risk: autorefresh causes excessive reruns.
  - Mitigation: user toggle + 10s default cadence.

## Open Questions
- Preferred default time range (24h vs 1h) for Overview KPIs.
- Which KPIs matter most for top-of-page hierarchy.
- Whether to expose advanced metrics panels by default or under expanders.

## Best-Practices Consulted
- docs/mcp_best_practices/README.md#L24-L77
- docs/mcp_best_practices/05-observability-telemetry.md
- docs/mcp_best_practices/12-timeout-resilience.md

