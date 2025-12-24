# Bikelane Intake Plan

This document defines the fast-intake ("bikelane") capability for foundry-mcp, aligned with the docs/mcp_best_practices and codebase standards.

## 1) Summary

Goal: Add a lightweight intake queue that captures work ideas quickly without forcing a full spec up front. Intake items live in a separate JSONL store and are exposed via new authoring actions (`intake-add`, `intake-list`). The bikelane is not a spec; specs remain the source of truth for implementation.

Key constraints:
- Storage lives at `specs/.bikelane/intake.jsonl`.
- `intake-list` returns only `status="new"` items.
- Use standard response envelopes (`response-v2`) and shared helpers.
- Use cursor-based pagination.

## 2) Goals

- Provide a fast, low-friction capture mechanism for incoming work.
- Keep intake data separate from spec lifecycle state to avoid validation and template complexity.
- Offer a concise, reliable list of new items for triage.
- Maintain strict input validation and predictable tool responses.

## 3) Non-Goals (for this phase)

- No automatic promotion from intake to spec (promotion can be a later phase).
- No triage or planning phases added to specs.
- No cross-linking between intake and specs.
- No background indexing, search, or tag analytics.

## 4) Storage

Location:
- Directory: `specs/.bikelane/`
- File: `specs/.bikelane/intake.jsonl`

Format:
- JSON Lines (one JSON object per line)
- Append-only writes for `intake-add`
- Read-only scan for `intake-list`
- No in-place mutation in v1

Rationale:
- JSONL is merge-friendly and supports append-only writes.
- Separate folder prevents spec validation tooling from treating intake data as a spec.

## 5) Intake Item Schema

Each line in `intake.jsonl` is an intake item with this schema:

```
{
  "schema_version": "intake-v1",
  "id": "intake-<uuid>",
  "title": "...",
  "description": "...",
  "status": "new",
  "priority": "p2",
  "tags": ["tag-a", "tag-b"],
  "source": "user-note",
  "requester": "tyler",
  "created_at": "2025-01-30T12:34:56Z",
  "updated_at": "2025-01-30T12:34:56Z"
}
```

Field rules:
- `schema_version`: Required. Fixed value `intake-v1` for v1.
- `id`: Required. Format `intake-<uuid4>`.
- `title`: Required. 1-140 chars.
- `description`: Optional. Max 2000 chars.
- `status`: Required. Fixed `new` in v1.
- `priority`: Optional. Enum `p0|p1|p2|p3|p4`. Default `p2`.
- `tags`: Optional. 0-20 items; each 1-32 chars; pattern `^[a-zA-Z0-9_-]+$`.
- `source`: Optional. Max 100 chars.
- `requester`: Optional. Max 100 chars.
- `created_at`/`updated_at`: Required. ISO 8601 UTC with `Z` suffix.

## 6) New Tool Actions

Router: `authoring` (per naming conventions and unified tool surface)

### 6.1 authoring(action="intake-add")

Purpose:
- Capture a new intake item and append to `intake.jsonl`.

Inputs:
- `title` (required, string)
- `description` (optional, string)
- `priority` (optional, string enum)
- `tags` (optional, array of strings)
- `source` (optional, string)
- `requester` (optional, string)
- `path` (optional, string) - workspace override

Behavior:
- Validate inputs (see Validation section).
- Resolve specs directory using existing helpers.
- Create `.bikelane` folder if missing.
- Append a single JSONL record with `status="new"`.
- Return the created item and `intake_path`.

Response `data`:
- `item`: the created intake item
- `intake_path`: absolute path to `intake.jsonl`

Idempotency:
- Non-idempotent (each call appends a new item). Documented in spec.

### 6.2 authoring(action="intake-list")

Purpose:
- List only `status="new"` intake items.

Inputs:
- `cursor` (optional, string)
- `limit` (optional, integer) default 50, max 200
- `path` (optional, string) - workspace override

Behavior:
- Validate inputs (cursor, limit).
- Read and parse `intake.jsonl` line-by-line.
- Filter items with `status="new"`.
- Apply cursor-based pagination.

Response `data`:
- `items`: array of intake items
- `intake_path`: absolute path to `intake.jsonl`
- `total_count`: optional (only if inexpensive to compute in implementation)

Pagination `meta.pagination`:
- `cursor`: opaque next cursor or null
- `has_more`: boolean
- `page_size`: integer

## 7) Pagination

- Cursor-based pagination only.
- Cursor encodes the last emitted item id and format version (opaque base64 JSON).
- `limit` is clamped to `[1..200]`.
- Invalid cursor returns `INVALID_FORMAT` with remediation.

Example cursor payload:
```
{"last_id": "intake-...", "version": 1}
```

## 8) Validation

Order: required fields, type checks, format checks, business rules.

Key rules:
- `title` required, non-empty, <= 140 chars.
- `description` optional, <= 2000 chars.
- `priority` enum; default `p2`.
- `tags` list length <= 20; each tag 1-32 chars and pattern `^[a-zA-Z0-9_-]+$`.
- `source`, `requester` length <= 100 chars.
- `limit` integer 1-200 (default 50).
- `cursor` must decode to expected payload.

## 9) Error Semantics

All errors use `error_response` with standard codes:
- `MISSING_REQUIRED` for missing `title`.
- `INVALID_FORMAT` for invalid `cursor`, non-string fields, or bad tag format.
- `VALUE_OUT_OF_RANGE` for lengths/limits.
- `NOT_FOUND` if specs directory cannot be resolved.
- `INTERNAL_ERROR` for IO failures.

Error responses include:
- `error_code`, `error_type`, `remediation`, `details.field`.

## 10) Security & Trust Boundaries

- Treat all inputs as untrusted (LLM-generated).
- Resolve specs directory via existing helpers to prevent path traversal.
- Avoid logging full description text (truncate or sanitize).
- Store only sanitized strings (strip control chars if needed).

## 11) Observability

- Add timing metrics for `authoring.intake-add` and `authoring.intake-list`.
- Include `telemetry.duration_ms` in responses.
- Audit log tool invocations with minimal payload (no full descriptions).

## 12) Feature Flags

- Add `intake_tools` feature flag (state: experimental).
- Gate the new actions behind this flag in both manifest and tool router.

## 13) Discovery / Manifest Updates

Update `mcp/capabilities_manifest.json`:
- Add `intake-add` and `intake-list` to `authoring` action enum.
- Add parameter definitions for `title`, `description`, `priority`, `tags`, `source`, `requester`, `cursor`, `limit`, `path`.
- Add examples for both actions.
- Add `intake_tools` in feature flags.

## 14) Specs, Docs, and Tests (Must Move Together)

Specs:
- Create a new spec JSON under `specs/pending/` that defines:
  - Inputs/outputs/errors for `intake-add` and `intake-list`
  - Pagination contract
  - Idempotency notes

Docs:
- Update relevant guides if they describe authoring actions or tool discovery.
- Add a brief section describing bikelane and how to list new items.

Tests:
- Unit tests for `intake-add` (validation, file creation, JSONL append).
- Unit tests for `intake-list` (empty file, pagination, invalid cursor).
- Fixture data for JSONL parsing and pagination.

## 15) Open Questions (Deferred)

- Promotion workflow from intake to spec.
- Status transitions beyond `new`.
- Tag search / filtering.
- Spec linkage or provenance in intake items.

## 16) Minimal Acceptance Criteria

- `authoring(intake-add)` writes a valid JSONL entry to `specs/.bikelane/intake.jsonl`.
- `authoring(intake-list)` returns only `status="new"` items with cursor-based pagination.
- All responses use response-v2 envelope with helpers.
- Inputs are validated with clear errors and remediations.
- Capability manifest and docs updated alongside code.
