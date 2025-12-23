# PLAN: Phase-First Spec Authoring Improvements

## Overview
Deliver three coordinated enhancements that reduce manual MCP calls without relying on Markdown parsing:
1. **Phase authoring macros** that create entire phases (plus starter tasks/placeholders) in a single command.
2. **Updated documentation** clarifying the supported schema creation/export workflow.
3. **Batch metadata utilities** so spec authors can update large sets of tasks/verify nodes at once.

## Goals & Non-Goals
- **Goals**
  - Let authors scaffold a whole phase (title, summary, default tasks) with one operation.
  - Align the sdd-plan documentation with the current set of `spec`/`authoring` commands.
  - Provide tooling to fix metadata-only validation issues through bulk operations.
- **Non-Goals**
  - Markdown/import parsing of plan documents.
  - Full redesign of task authoring flows; individual task commands still exist for fine tuning.
  - GUI experiences or IDE plugins.

## Success Metrics
- A complete phase can be added with <=2 commands (phase macro + optional metadata tweak).
- Skill docs contain no references to deprecated actions and include copy/pasteable workflows.
- Batch metadata command can fix all verification/file_path errors in <3 commands even on large specs.

## Work Breakdown

### 1. Phase Authoring Macros (High Priority)
- **API Design**
  - Extend authoring plugin with `phase-add-bulk` that accepts a JSON payload describing phase name, summary, and an ordered list of task/verify placeholders.
  - Allow reuse of predefined phase templates (e.g., “Labeling App Phase 1”) stored with the skill.
- **Implementation Steps**
  - Add server-side handler that creates the phase node and loops through provided child definitions, creating tasks/verify nodes atomically.
  - Support optional metadata defaults (e.g., default owner, spec section) applied to all generated tasks.
  - Provide dry-run output so authors can inspect the structure before creation.
- **Skill Integration**
  - Update sdd-plan commands to surface macros (e.g., `phase add template 1`, `phase add custom` with inline JSON editor).
  - Include helper snippets for copying a plan section and pasting it into the macro payload.
- **Testing**
  - Unit coverage for payload validation (missing titles, duplicated IDs).
  - Integration test that creates phases 1–3 of the labeling spec using only macro calls.
- **Example Tool Calls**
  ```bash
  mcp__plugin_foundry_foundry-mcp__authoring phase-add-bulk '{"spec_id":"spec-ai-labeling-001","phase":{"phase_id":"phase-3","title":"Integrate Ingestion","summary":"Hook labelers into scoring app"},"children":[{"type":"task","title":"Wire ingestion API","metadata":{"file_path":"src/ingestion/api.py","estimated_hours":3}},{"type":"verify","title":"Regression suite","metadata":{"verification_type":"run-tests","mcp_tool":"mcp__foundry-mcp__test-run"}}]}'
  mcp__plugin_foundry_foundry-mcp__authoring phase-add-template '{"spec_id":"spec-ai-labeling-001","template_id":"labeling-phase-1","phase_id":"phase-1"}'
  ```

### 2. Documentation Update for Schema Export Flow (Medium Priority)
- **Audit** existing `spec`/`authoring` features to ensure docs reflect reality.
- **Doc Changes**
  - Remove references to `spec schema-export` in `skills/sdd-plan/SKILL.md` and related references.
  - Add a “Phase-first authoring” section walking through: create spec from template → run phase macro → fine tune tasks → run metadata batch.
  - Highlight fallback commands (manual `task add`) for edge cases.
- **Changelog & Guidance**
  - Document the rationale for retiring schema-export and link to the new macro workflow in troubleshooting guides.
- **Example Tool Calls**
  ```bash
  mcp__plugin_foundry_foundry-mcp__authoring spec-create '{"name":"spec-ai-labeling-001","template":"medium","category":"implementation"}'
  mcp__skill_sdd-plan doc-open 'phase-first-authoring'
  ```

### 3. Batch Metadata Utilities (High Priority)
- **Command Surface**
  - Introduce `task metadata-batch` that targets nodes by phase ID, task name regex, or node type.
  - Support multiple metadata keys per invocation (`file_path`, `verification_type`, owners, labels).
- **Verification Fixers**
  - Provide canned operations, e.g., `fix-verification-types phase=verify` to rewrite enums to `test/fidelity` values.
  - Ensure validator and metadata API share a single source of truth for allowed values.
- **Safety & UX**
  - Add dry-run flag and summary output (count of nodes updated, diff preview).
  - Support rollback on partial failure (transaction or revert copy of previous metadata).
- **Testing**
  - Regression tests confirming bulk updates succeed on 50+ tasks and gracefully handle invalid filters.
- **Example Tool Calls**
  ```bash
  mcp__plugin_foundry_foundry-mcp__task metadata-batch '{"spec_id":"spec-ai-labeling-001","filter":{"phase_id":"phase-4","node_type":"task"},"metadata":{"file_path":"src/labeling/phase4.py"}}'
  mcp__plugin_foundry_foundry-mcp__task metadata-batch '{"spec_id":"spec-ai-labeling-001","filter":{"node_type":"verify"},"metadata":{"verification_type":"fidelity"},"dry_run":true}'
  mcp__plugin_foundry_foundry-mcp__task metadata-batch '{"spec_id":"spec-ai-labeling-001","filter":{"phase_id":"phase-7","node_type":"verify"},"metadata":{"verification_type":"run-tests"}}'
  ```

## Dependencies & Risks
- **Dependencies**
  - MCP authoring/task routes that can accept batch payloads.
  - Maintained phase templates stored in the repo/skill.
- **Risks**
  - Large payloads might hit RPC limits; mitigate with pagination or per-phase batching.
  - Batch metadata ops could overwrite intentional differences; dry-run and filter validation are critical.
  - Documentation must stay in sync as new macros ship; add owner to review doc updates alongside code changes.

## Timeline (Rough)
1. **Week 1**: Finalize phase macro payload schema; stub batch metadata command; draft doc updates.
2. **Week 2**: Implement phase macros + tests; build metadata-batch backend with dry-run support.
3. **Week 3**: Integrate macros into sdd-plan skill, update documentation, run end-to-end scenario tests.

## Validation & Rollout
- Dogfood on the labeling spec workflow to confirm each phase can be authored in a single macro call plus metadata fix.
- Update release notes + sdd-plan changelog; provide short Loom/demo if useful.
- Track telemetry (MCP call counts per spec) to confirm reduction after rollout.
