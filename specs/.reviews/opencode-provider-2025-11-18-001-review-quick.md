# Specification Review Report

**Spec**: Specification (`opencode-provider-2025-11-18-001`)
**Review Type**: Quick
**Date**: 2025-11-18 18:40:00
**Models Consulted**: 2 (gemini, claude)

---

# Synthesis

## Overall Assessment
- **Consensus Level**: Strong
- **Summary**: Both models agree that the architectural pattern (Node.js wrapper) is sound and consistent with existing providers, but the specification is currently unimplementable due to critical gaps. Gemini focuses on runtime execution issues (streaming/buffering, error propagation), while Claude focuses on lifecycle issues (installation, binary detection, API definitions). Both highlight that the `tool` and `session` interfaces are under-specified.

## Critical Blockers
Issues that must be fixed before implementation (identified by multiple models):

- **[Architecture]** Streaming vs. Buffering Mismatch - flagged by: gemini
  - **Impact**: The spec asks Node.js to output a single JSON response at the end, but asks Python to stream. This will cause the UI to hang until generation is complete, breaking the "typing" effect.
  - **Recommended fix**: Redesign the Node.js wrapper to output NDJSON (Newline Delimited JSON) events to stdout, and update the Python parser to handle this stream.

- **[Architecture]** Binary Availability & Detection Logic - flagged by: claude
  - **Impact**: The detection logic looks for an `opencode` binary, but the implementation uses `node` + a local wrapper script. `is_opencode_available()` will always return false, preventing usage.
  - **Recommended fix**: Change detection strategy to check for `node` presence and the existence of the wrapper script, rather than a standalone `opencode` binary.

- **[Completeness]** Missing SDK API & Dependency Specs - flagged by: claude
  - **Impact**: The specific API calls (`createOpencode`, `session.prompt`) and response schemas are referenced but not documented. Additionally, there is no instruction on how `npm install` occurs for a Python package user.
  - **Recommended fix**: Add an "SDK Reference" section to the spec and define the lifecycle hook for installing Node dependencies (e.g., via `setup.py` or documentation).

- **[Completeness]** Phase 5 (Documentation) is Empty - flagged by: gemini
  - **Impact**: Phase 5 claims to have tasks but the file list is empty.
  - **Recommended fix**: Add specific tasks to update `docs/AI_CONTEXT.md` and `docs/DOCUMENTATION.md`.

## Major Suggestions
Significant improvements that enhance quality, maintainability, or design:

- **[Error Handling]** Stderr Propagation - flagged by: gemini
  - **Description**: The spec doesn't define how Node.js runtime errors are passed to Python.
  - **Recommended fix**: Explicitly mandate that the Python wrapper captures `stderr` from the subprocess and includes it in exceptions.

- **[Interface Design]** Tool Restriction Format - flagged by: claude, gemini
  - **Description**: The format for `--allowedTools` is undefined, and it is unclear if the SDK accepts `sdd-toolkit` tool formats natively.
  - **Recommended fix**: Define the exact CLI argument format (e.g., CSV) and the mapping logic between Python tool names and OpenCode SDK tool identifiers.

- **[Data Model]** Model Descriptor Configuration - flagged by: claude
  - **Description**: The spec mentions "user-configurable" models but lists no defaults or supported IDs.
  - **Recommended fix**: Add a section documenting supported OpenCode model IDs (e.g., `sonnet-4-5`) and default priorities.

## Questions for Author
Clarifications needed (common questions across models):

- **[Security]** Authentication Mechanism - flagged by: gemini
  - **Context**: How does the Node.js SDK authenticate? Does it need an API key passed via `env` or does it read a system variable?

- **[Architecture]** Session Persistence & State - flagged by: claude
  - **Context**: Does `createOpencode()` start a persistent server? Should the wrapper script stay alive across prompts, or is it one-shot?

- **[Interface]** Session Title & System Prompts - flagged by: claude
  - **Context**: The spec requires a "title" and "system prompt" but doesn't explain their source or if they persist across the session.

## Design Strengths
What the spec does well (areas of agreement):

- **[Architecture]** Language Isolation (Wrapper Pattern) - noted by: gemini, claude
  - **Why**: Using a subprocess to bridge Python and Node.js is pragmatic, avoids complex FFI, and matches existing project patterns (making it easier to maintain).
- **[Process]** Fidelity Reviews - noted by: claude
  - **Why**: The inclusion of strict fidelity verification at the end of every phase is a strong quality assurance measure.

## Points of Agreement
- The wrapper pattern is the correct architectural choice.
- The specification for "Tools" is too vague to implement.
- Documentation tasks (Phase 5) are missing or incomplete.

## Points of Disagreement
- **Focus on Dependencies**: Claude is very concerned with how `node_modules` are installed/distributed in a Python environment. Gemini did not flag this as a blocker (though noted versioning), focusing instead on the runtime data flow (streaming).
- **Assessment**: Both points are valid. The "Dependency" issue is a deployment blocker, while the "Streaming" issue is a UX blocker. Both must be addressed.

## Synthesis Notes
- **Actionable Next Steps**:
    1.  Rewrite the **Architecture** section to define the Streaming Interface (NDJSON) and Detection Logic (Node+Script).
    2.  Add a **Technical Reference** section detailing the OpenCode SDK methods and inputs.
    3.  Clarify the **Installation Lifecycle** (how `npm install` relates to `pip install`).
    4.  Fill in the empty **Phase 5** tasks.

---

## 📝 Individual Model Reviews

### gemini

# Review Summary

## Critical Blockers
Issues that must be fixed before implementation can begin.

- **[Completeness]** Phase 5 File Modifications Empty
  - **Description:** The `phase-5-files` group claims `total_tasks: 1` but has an empty `children` array. There are no actual tasks defined for documentation updates.
  - **Impact:** The "Documentation updates" mentioned in the phase description will not be tracked or implemented, leading to undocumented features.
  - **Fix:** Add specific tasks to Phase 5 to update `docs/AI_CONTEXT.md`, `docs/DOCUMENTATION.md`, or create a new `docs/providers/opencode.md`.

- **[Architecture]** Streaming Implementation Mismatch
  - **Description:** Task 1-1-4 describes the Node.js wrapper formatting a single "JSON response" at the end of execution. However, Task 2-1-4 describes the Python provider as "emit streaming chunks".
  - **Impact:** Real-time token streaming (typing effect) will not work. The Python provider will block until the Node process finishes, then emit everything at once, degrading user experience.
  - **Fix:** Redesign the Node.js wrapper to output Newline Delimited JSON (NDJSON) events for tokens if the SDK supports streaming, and update Task 2-1-4 to parse this stream. Alternatively, explicitly mark the provider as non-streaming.

## Major Suggestions
Significant improvements that enhance quality, maintainability, or design.

- **[Error Handling]** Stderr Propagation
  - **Description:** The spec doesn't explicitly detail how Node.js runtime errors (stderr) are propagated to the Python provider's exception handling.
  - **Impact:** Debugging failures (like missing dependencies or SDK errors) will be difficult if stderr is swallowed or genericized.
  - **Fix:** In Task 2-1-4, ensure the subprocess execution captures `stderr` and includes it in the `ProviderExecutionError` message if the exit code is non-zero.

## Minor Suggestions
Smaller improvements and optimizations.

- **[Verification]** Explicit Dependency Versioning
  - **Description:** Task 1-2-1 mentions adding `@opencode-ai/sdk` but doesn't specify a version.
  - **Fix:** Specify a minimum version number to ensure API compatibility with the methods (`createOpencode`, `session.prompt`) relied upon in the implementation.

## Questions
Clarifications needed or ambiguities to resolve.

- **[Security]** Authentication Mechanism
  - **Context:** The spec mentions `createOpencode()` and `session` management but is silent on API Keys.
  - **Needed:** How does the Node.js SDK authenticate? Does it read a specific environment variable (e.g., `OPENCODE_API_KEY`) automatically, or does the Python provider need to pass this explicitely in the `env` dictionary during subprocess creation?

- **[Interface Design]** Tool Definition Format
  - **Context:** Task 1-1-3 mentions "tool restrictions via --allowedTools".
  - **Needed:** Does the OpenCode SDK accept tool definitions in the same format as the `claude-sdd-toolkit`'s internal representation, or is a transformation step required in the Python provider before passing them to the wrapper?

## Praise
What the spec does well.

- **[Architecture]** Language Isolation
  - **Why:** Using a subprocess wrapper to bridge the Python application with a Node.js-only SDK is a pragmatic solution that avoids complex foreign function interfaces (FFI) while maintaining process isolation.

---

### claude

# Review Summary

## Critical Blockers

- **[Architecture]** OpenCode CLI binary availability and detection inconsistency
  - **Description:** The spec assumes an `opencode` CLI binary exists for detection (task-3-2-1, verify-3-2) and is used as `DEFAULT_BINARY` for detection (detectors.py), but Phase 1 uses the Node.js SDK via `opencode_wrapper.js` which requires `node` binary. The actual OpenCode execution path doesn't use an `opencode` CLI binary at all.
  - **Impact:** Provider detection will fail. `is_opencode_available()` will look for a binary that doesn't exist in the execution flow. Users won't be able to use the provider even after correct installation.
  - **Fix:** Clarify detection strategy - either (1) detect `node` binary + verify Node.js wrapper script exists, or (2) if OpenCode does provide a separate CLI binary, explain how it relates to the Node.js SDK wrapper approach and when each is used.

- **[Completeness]** Missing OpenCode SDK API specification
  - **Description:** The spec references `createOpencode()`, `session.prompt()`, and response structure with `parts` array and `info` object, but doesn't document the actual API contract, required/optional parameters, or response schema from @opencode-ai/sdk.
  - **Impact:** Implementer cannot write `opencode_wrapper.js` without external documentation. Response parsing in Python (task-2-1-4) may be incorrect if response structure is misunderstood.
  - **Fix:** Add an "OpenCode SDK Reference" section documenting: (1) `createOpencode()` parameters and return value, (2) session creation API and parameters, (3) `session.prompt()` signature and options, (4) exact response schema with `parts` array structure and `info.tokenUsage` fields.

- **[Architecture]** Node.js dependency installation location and lifecycle unclear
  - **Description:** Task-1-4 installs npm dependencies in `src/claude_skills/claude_skills/common/providers/`, but spec doesn't address: (1) how this integrates with Python package installation (setuptools/pip), (2) whether `node_modules/` should be in `.gitignore`, (3) installation instructions for end users.
  - **Impact:** Provider won't work after `pip install` unless users manually run `npm install` in a nested directory. Deployment and distribution will fail.
  - **Fix:** Add to Phase 1 or create Phase 0: (1) update Python packaging (`setup.py`/`pyproject.toml`) to run `npm install` post-install, or (2) document manual installation steps in README, or (3) vendor the SDK if licensing allows. Specify whether `node_modules/` is distributed or .gitignored.

## Major Suggestions

- **[Interface Design]** Tool restrictions implementation mechanism undefined
  - **Description:** Task-1-1-3 mentions "tool restrictions via --allowedTools CLI arg" but doesn't specify: (1) format of this argument (comma-separated? JSON?), (2) how Python provider maps its tool restriction list to this format, (3) what tools are available/how they're named in OpenCode.
  - **Impact:** Tool restriction feature may not work correctly. Implementation will require trial-and-error or external documentation lookup.
  - **Fix:** Specify exact format: "Add `--allowedTools` argument taking comma-separated tool names matching OpenCode SDK tool identifiers (e.g., `--allowedTools Read,Write,Bash`). Map Python tool names to OpenCode equivalents in task-2-1-4."

- **[Data Model]** Model descriptor configuration strategy ambiguous
  - **Description:** Task-2-1-2 says "user-configurable via ai_config.yaml" and task-3-3-1 says "Add opencode to DEFAULT_MODELS with priority list (user-configurable)" but doesn't specify: (1) what model IDs OpenCode supports, (2) default model list, (3) model capability differences.
  - **Impact:** Implementer will hard-code arbitrary model names. Users won't know what models are available or how to configure them.
  - **Fix:** Add section documenting: (1) supported OpenCode model IDs (e.g., "sonnet-4-5", "opus-4"), (2) default priority list for OPENCODE_MODELS, (3) capability matrix if models differ in vision/tool support.

- **[Verification]** Insufficient error scenario coverage in testing
  - **Description:** Task-4-1-3 tests basic errors (binary not found, timeout, invalid JSON) but doesn't test: (1) OpenCode SDK initialization failures, (2) session creation errors, (3) rate limiting responses, (4) partial/streaming response errors, (5) tool execution failures within OpenCode.
  - **Impact:** Production issues won't be caught by tests. Error messages will be poor. Debugging will be difficult.
  - **Fix:** Expand task-4-1-3 to include: "Test SDK connection errors (server unreachable), session quota exceeded, tool execution denied, malformed streaming chunks, and verify error messages provide actionable guidance."

## Minor Suggestions

- **[Architecture]** Working directory parameter handling inconsistent
  - **Description:** Task-1-1-1 mentions "working directory" as CLI argument, task-1-1-2 says "create session with title and working directory", but no task specifies how Python provider obtains/sets working directory or what the default should be.
  - **Fix:** In task-2-1-4, specify: "Pass `cwd=os.getcwd()` or workspace root from provider context as `--workingDir` argument. Default to current process working directory if not specified in GenerationRequest."

- **[Security]** Security flags rationale not documented
  - **Description:** Task-2-1-2 sets `writes_allowed: False` in OPENCODE_METADATA but spec doesn't explain why or how this compares to other providers (Gemini/Codex settings).
  - **Fix:** Add comment: "Set writes_allowed: False initially for safety - OpenCode SDK tools are sandboxed. Update to True in future after validating tool restriction enforcement."

- **[Verification]** verify-1-3 test insufficient
  - **Description:** Verification only checks valid JSON output but doesn't validate the actual fields required by Python provider (text, tokens, model).
  - **Fix:** Change verify-1-3 expected: "Valid JSON response with 'content' or 'parts' field containing text, 'info.tokenUsage' object with input/output counts, and 'model' field."

## Questions

- **[Architecture]** Does OpenCode SDK maintain persistent server/session state?
  - **Context:** If `createOpencode()` starts a persistent server process, wrapper script should reuse it across invocations rather than starting/stopping per request. Affects performance and resource usage.
  - **Needed:** Clarify if: (1) server persists between script invocations, (2) session state is preserved, (3) cleanup/shutdown is needed, (4) concurrent request handling is supported.

- **[Interface Design]** How does system prompt interact with session configuration?
  - **Context:** Task-1-1-1 mentions "system prompt" as CLI arg, task-1-1-3 says "call session.prompt() with system prompt". Unclear if system prompt is per-session or per-prompt, and whether it affects session creation.
  - **Needed:** Specify: (1) where system prompt is configured (session creation vs prompt call), (2) whether it persists across multiple prompts in same session, (3) if it's optional or required.

- **[Data Model]** What is the "title" parameter for session creation?
  - **Context:** Task-1-1-2 says "create session with title" but doesn't specify what this title represents or how to generate it.
  - **Needed:** Explain what session title is used for (logging? UI display?) and provide generation strategy (e.g., timestamp, request ID, user-provided).

- **[Verification]** What constitutes acceptable performance for verify-5-1?
  - **Context:** End-to-end test should succeed, but no performance baseline specified. If request takes 60+ seconds, is that acceptable?
  - **Needed:** Add performance expectations: "Response should complete within 30 seconds for simple prompt. Token usage should be non-zero and reasonable (< 1000 for '2+2' query)."

## Praise

- **[Architecture]** Excellent pattern reuse from existing providers
  - **Why:** Following Gemini/Codex patterns (task-2-1, Phase 2) ensures consistency with codebase conventions, makes review easier, and reduces integration risks. Clear reference to existing implementations accelerates development.

- **[Verification]** Comprehensive fidelity reviews at each phase
  - **Why:** Phase-end fidelity reviews (verify-1-4, verify-2-1, verify-3-3, verify-4-2, verify-5-2) with "continue_on_failure: false" ensure high quality and catch drift early. This structured approach prevents accumulation of technical debt.

- **[Completeness]** Well-structured phase dependencies
  - **Why:** Clear blocking relationships between phases (1→2→3→4→5) prevent premature integration and ensure solid foundation before building higher layers. Granular task dependencies within phases enable parallel work where possible.

---
