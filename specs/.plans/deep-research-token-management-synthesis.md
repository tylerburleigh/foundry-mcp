# Synthesis

## Overall Assessment
- **Consensus Level**: Moderate (based on agreement across models)

Both reviewers identify critical blockers, but they focus on different aspects of the same underlying concerns. There is strong agreement on the need for improvements around runtime overhead configuration and protected content handling, but the reviewers prioritize different architectural risks.

## Critical Blockers
Issues that must be fixed before implementation (identified by multiple models):

- **[Architecture]** Warning schema inconsistency between `meta.warnings` and `data.warning_details` - flagged by: cursor-agent
  - **Impact**: Client handling becomes inconsistent; fixtures/tests may become brittle; downstream tools may miss warnings or treat absence as error.
  - **Recommended fix**: Define a single canonical warning contract: either always include both with empty defaults or only emit both together. Document explicit behavior for no-warning cases and ensure fixture expectations align.

- **[Architecture]** Token counting strategy prioritizes network-dependent APIs - flagged by: gemini
  - **Impact**: Significant slowdown of the research loop; potential for rate-limit exhaustion solely from counting tokens.
  - **Recommended fix**: Change priority to: **Local Tokenizer (e.g., tiktoken, google-generativeai local)** > **Heuristic (char/4)**. Only use provider API counters if a local option is strictly impossible and the operation is infrequent (e.g., final pre-flight check, not iterative budgeting).

- **[Completeness]** No explicit plan for provider token counting availability - flagged by: cursor-agent
  - **Impact**: Implementation risk is high; preflight logic may be unreliable or inconsistent across providers, leading to overflow regressions.
  - **Recommended fix**: Add a provider capability matrix (per provider: native count, tokenizer, heuristic only), define an interface in `ProviderManager`, and specify fallback behavior for each provider.

- **[Risk]** Content archive privacy boundary not fully defined despite "out of scope" - flagged by: cursor-agent
  - **Impact**: Potential leakage of sensitive data into disk archives, which conflicts with a security-first posture even if governance is "out of scope."
  - **Recommended fix**: Add a minimal safety gate: allow per-item `archive_allowed` or check `ContentItem.sensitive` to skip archival and emit a warning. This is a minimal risk mitigation without defining full governance.

## Major Suggestions
Significant improvements that enhance quality, maintainability, or design:

- **[Feasibility]** `runtime_overhead` configuration risk - flagged by: cursor-agent, gemini
  - **Description**: Both reviewers identify that relying on user-configured static values is error-prone. cursor-agent notes lack of validation/sanity checks; gemini suggests implementing a calibration command to measure actual overhead dynamically.
  - **Recommended fix**: 
    - Add minimum/maximum validation (e.g., 0 ≤ overhead ≤ context_window) and an informational warning when overhead exceeds a threshold (e.g., >50% of context).
    - Consider implementing a "Calibration" command (or startup routine) that measures the *actual* system prompt + tool definition size using the configured tokenizer and suggests (or sets) the `runtime_overhead` value dynamically.

- **[Completeness]** Protected content detection logic not specified - flagged by: cursor-agent, gemini
  - **Description**: cursor-agent notes the plan relies on `ContentItem.protected: bool` but doesn't specify how it's determined; gemini asks how "key findings" are marked as protected.
  - **Recommended fix**: Define the logic or heuristic for setting `protected=True` (e.g., "Items explicitly referenced in a 'Findings' list", "Sources with high relevance scores (>0.9)", or "User-pinned items").

- **[Architecture]** Map-reduce budget & "Reduce" logic missing - flagged by: gemini
  - **Description**: The plan mentions "Map-reduce" for summarization but doesn't define the "Reduce" strategy. If "Reduce" involves another LLM call to synthesize chunks, that call itself consumes budget and time.
  - **Recommended fix**: Explicitly define the "Reduce" strategy (concatenation vs. synthesis). If synthesis, reserve a specific token buffer for the final combination prompt and include it in the `SummarizationResult` cost calculation.

- **[Architecture]** Model limit defaults are hardcoded and may be stale - flagged by: cursor-agent
  - **Description**: `DEFAULT_MODEL_LIMITS` is a static dict with values that will drift. There is no plan for updates or validation.
  - **Recommended fix**: Add a versioned source-of-truth doc or fixture, plus a validation test ensuring each provider's limits can be overridden and are documented with a date/source.

- **[Sequencing]** Contract and fixture gating lacks explicit test order - flagged by: cursor-agent
  - **Description**: The gate says update contracts/fixtures before implementation, but no explicit order for tests or fixture regeneration is defined.
  - **Recommended fix**: Add a sequencing step: update schema docs → update fixtures → run schema validation tests → implement logic → update integration tests.

- **[Risk]** Summarization fallback loop could amplify costs - flagged by: cursor-agent
  - **Description**: The plan retries per provider and re-summarizes to tighter levels without a maximum cost budget or token/call limit.
  - **Recommended fix**: Add a per-item summarization budget cap (max tokens or max calls) and record "summary_truncated_due_to_budget" warnings.

- **[Completeness]** No explicit mapping for how `content_fidelity` merges across multiple phases - flagged by: cursor-agent
  - **Description**: The plan says "latest phase overwrites same-phase entry," but doesn't define how the aggregate `content_fidelity` is scoped per phase or per iteration.
  - **Recommended fix**: Add explicit structure: `content_fidelity[item_id].phases[phase][iteration]` or store `last_updated` + `iteration` and define a merge rule.

## Questions for Author
Clarifications needed (common questions across models):

- **[Clarity]** How are phase budgets split between items and the system prompt payload? - flagged by: cursor-agent
  - **Context**: The budgeting logic counts content items but does not clearly include system prompt and tool schema payloads.
  - **Needed**: Explicit accounting rules for fixed overhead vs dynamic message overhead.

- **[Architecture]** How does `content_fidelity` handle merged or derived items? - flagged by: cursor-agent
  - **Context**: Summaries produced from multiple sources need fidelity tied to multiple `source_ids`.
  - **Needed**: A mapping strategy for summaries that blend sources (e.g., one summary with multiple `source_ids`).

- **[Feasibility]** What is the exact "preflight_count" API and where is it implemented? - flagged by: cursor-agent
  - **Context**: Several phases depend on `preflight_count(payload, provider_id)` but no signature or integration location is specified.
  - **Needed**: Define interface, return values, and which layer owns it.

- **[Risk]** How are warnings deduplicated across phases? - flagged by: cursor-agent
  - **Context**: The plan can emit warnings at per-item, per-phase, and response-level.
  - **Needed**: Rule for deduplication/aggregation to avoid noisy `meta.warnings`.

- **[Completeness]** Will token management be applied to all deep research tools or only the main workflow? - flagged by: cursor-agent
  - **Context**: The contract matrix focuses on `deep-research-report`.
  - **Needed**: Clarify whether status/list endpoints expose new fidelity metadata or remain unchanged, and if any other tools will include warnings.

- **[Sequencing]** Migration trigger mechanism - flagged by: gemini
  - **Context**: The plan mentions "Run migrations on state load."
  - **Needed**: Clarification on where this hook exists. Is there an existing `DeepResearchState.load()` method that intercepts the raw JSON before validation? If not, `pydantic` validation might fail on schema version mismatches before migration logic runs.

## Design Strengths
What the spec does well (areas of agreement):

- **[Architecture]** Simplification / YAGNI approach - noted by: cursor-agent, gemini
  - **Why**: Both reviewers praise the choice of static dictionaries (`DEFAULT_MODEL_LIMITS`) and manual overrides instead of complex dynamic calibration. It makes the system predictable and easier to debug for a CLI tool.

- **[Architecture]** Proactive budgeting and provider-aware limits - noted by: cursor-agent
  - **Why**: It addresses known overflow failure modes and aligns to real-world provider differences.

- **[Clarity]** Fidelity & warning taxonomy - noted by: gemini
  - **Why**: The breakdown of warning codes (`CONTENT_TRUNCATED`, `PROTECTED_OVERFLOW`) and the explicit `content_fidelity` schema provide excellent visibility into *what* the system did to the user's data, addressing the "silent failure" problem effectively.

- **[Design]** Combined vs. input-only budgeting - noted by: gemini
  - **Why**: Distinguishing between models that share input/output limits (OpenAI/Anthropic) and those with separate buckets (Gemini) is a crucial detail that prevents "budget math" errors.

- **[Sequencing]** Contract/fixtures gate before implementation - noted by: cursor-agent
  - **Why**: It reduces integration risk and ensures schema discipline.

- **[Risk]** Graceful degradation hierarchy is well structured - noted by: cursor-agent
  - **Why**: Summarize → truncate → drop with warnings makes failure modes observable and recoverable.

- **[Completeness]** Coverage of tests is thorough - noted by: cursor-agent
  - **Why**: The verification checklist touches unit, integration, and schema propagation paths.

## Points of Agreement

- **Runtime overhead configuration is risky**: Both reviewers identify that user-configured `runtime_overhead` values are error-prone and need validation or calibration mechanisms.
- **Protected content detection needs specification**: Both reviewers note that the `protected` flag logic is undefined and needs explicit rules.
- **YAGNI approach is appropriate**: Both reviewers praise the simplification choices and static configuration approach.
- **Token counting strategy needs refinement**: Both reviewers identify issues with the token counting approach, though from different angles (cursor-agent focuses on provider availability, gemini focuses on network dependency).
- **Fidelity tracking is valuable**: Both reviewers acknowledge the value of explicit fidelity metadata and warning taxonomy.

## Points of Disagreement

- **Token counting priority**: 
  - **cursor-agent** focuses on the lack of a provider capability matrix and explicit fallback behavior per provider.
  - **gemini** focuses on the performance risk of network-dependent token counting APIs and recommends prioritizing local tokenizers.
  - **Assessment**: These are complementary concerns. The spec should address both: define provider capabilities AND prioritize local tokenizers to avoid network calls in hot paths.

- **Critical blocker prioritization**:
  - **cursor-agent** identifies 3 critical blockers (warning schema, provider token counting, archive privacy).
  - **gemini** identifies 1 critical blocker (network-dependent token estimation).
  - **Assessment**: All identified blockers are valid. The warning schema inconsistency and token counting strategy (both network dependency and provider availability) should be addressed before implementation. Archive privacy is a valid security concern but may be acceptable as a "minimal safety gate" rather than full governance.

## Synthesis Notes

### Overall Themes

1. **Token Counting Strategy Needs Clarification**: Both reviewers identify issues with token counting, but from different perspectives. The spec should:
   - Prioritize local tokenizers over network APIs for performance
   - Define a provider capability matrix for fallback behavior
   - Specify the exact `preflight_count` API signature and location

2. **Configuration Validation is Critical**: The `runtime_overhead` configuration is flagged by both reviewers as high-risk. The spec should add:
   - Validation bounds (min/max)
   - Warning thresholds
   - Optional calibration mechanism

3. **Protected Content Logic Must Be Defined**: Both reviewers note that `protected` flag determination is unspecified. The spec should explicitly define:
   - How items are marked as protected
   - What criteria determine protection
   - How protection interacts with summarization/dropping logic

4. **Warning Schema Consistency**: cursor-agent identifies a critical inconsistency between `meta.warnings` (omitted when empty) and `data.warning_details` (empty list when no warnings). This needs resolution before implementation.

5. **Map-Reduce Budget Accounting**: gemini identifies that the "Reduce" step budget is not accounted for. If synthesis is used, this needs explicit token reservation.

### Actionable Next Steps

1. **Immediate (Before Implementation)**:
   - Resolve warning schema inconsistency (always include both with empty defaults OR document explicit omission rules)
   - Define token counting priority: Local Tokenizer > Heuristic > Provider API (only for infrequent pre-flight)
   - Add provider capability matrix for token counting
   - Specify `preflight_count` API signature and implementation location
   - Define protected content detection logic

2. **High Priority (During Implementation)**:
   - Add `runtime_overhead` validation (bounds + warnings)
   - Implement map-reduce budget accounting for "Reduce" step
   - Add content fidelity merge rules for multi-phase/iteration scenarios
   - Define warning deduplication strategy across phases
   - Add minimal archive privacy gate (per-item `archive_allowed` or `sensitive` check)

3. **Medium Priority (Before Release)**:
   - Add versioned source-of-truth for model limits
   - Define explicit test sequencing for contract/fixture updates
   - Add summarization budget caps to prevent cost amplification
   - Clarify migration trigger mechanism (where `DeepResearchState.load()` intercepts)
   - Document which tools expose fidelity metadata (beyond `deep-research-report`)

4. **Documentation**:
   - Clarify phase budget split between items and system prompt payload
   - Document how `content_fidelity` handles merged/derived items
   - Specify archive permissions enforcement (`os.chmod(path, 0o600)`)
   - Define chunk ID format for `content_fidelity` keys

### Consensus Areas to Preserve

- Keep the YAGNI simplification approach (static config, manual overrides)
- Maintain the graceful degradation hierarchy (summarize → truncate → drop)
- Preserve the contract/fixtures gate before implementation
- Continue with provider-aware budgeting (combined vs input-only)
- Keep the comprehensive test coverage plan
