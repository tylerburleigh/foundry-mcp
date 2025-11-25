# Doc-Query CLI Enhancement

**Spec ID:** `doc-query-enhancements-2025-10-24-001`  
**Status:** pending (0/46 tasks, 0%)  
**Estimated Effort:** 230 hours  
**Complexity:** high  

Enhance doc-query skill with workflow automation, cross-reference tracking, and improved UX to reduce manual command chaining from 6-8 steps to 1 step for common workflows

## Objectives

- Fix critical reverse dependency support
- Add cross-reference data (callers/callees) to schema v2.0
- Implement 4 core wrapper commands reducing workflows by 80%
- Add native pattern filtering to eliminate grep dependencies
- Improve impact analysis accuracy from ~60% to ~95%

## P0: Foundation - Critical Fixes (0/21 tasks, 0%)

**Purpose:** Fix critical blockers: reverse dependency support and cross-reference generation  
**Risk Level:** high  
**Estimated Hours:** 40  


### File Modifications (0/16 tasks)

#### ⏳ Verify reverse dependency support

**File:** `investigation`  
**Status:** pending  
**Estimated:** 4 hours  

##### ⏳ Inspect documentation.json schema for reverse dependency data

**Status:** pending  

##### ⏳ Test sdd doc dependencies --reverse on real codebase

**Status:** pending  

**Blocked by:** task-1-1-1

##### ⏳ Document findings and determine if fix needed

**Status:** pending  

**Blocked by:** task-1-1-2

#### ⏳ src/claude_skills/claude_skills/code_doc/ast_analysis.py

**File:** `src/claude_skills/claude_skills/code_doc/ast_analysis.py`  
**Status:** pending  
**Estimated:** 16 hours  

**Blocked by:** task-1-1

##### ⏳ Implement function call tracking during AST traversal

**Status:** pending  
**Changes:** ['Add call tracking in AST visitor', 'Capture function call expressions', 'Store caller/callee relationships', 'Track call locations (file, line)']  

##### ⏳ Implement import tracking for class instantiation

**Status:** pending  
**Changes:** ['Track import statements', 'Capture class instantiation points', 'Record imported_by relationships', 'Store instantiated_by data']  

##### ⏳ Build bidirectional reference graph

**Status:** pending  
**Changes:** ['Create caller/callee graph structure', 'Maintain bidirectional references', 'Support reverse lookups']  

**Blocked by:** task-1-2-1, task-1-2-2

##### ⏳ Add warning system for dynamic Python patterns

**Status:** pending  
**Changes:** ['Detect decorators, monkey-patching, and reflection during AST analysis', 'Log warnings when dynamic patterns are found that may affect cross-reference accuracy', 'Document limitations in generated documentation.json metadata', 'Add coverage statistics for dynamic vs static call detection']  

#### ⏳ src/claude_skills/claude_skills/code_doc/schema.py

**File:** `src/claude_skills/claude_skills/code_doc/schema.py`  
**Status:** pending  
**Estimated:** 8 hours  

**Blocked by:** task-1-2

##### ⏳ Add callers and calls fields to function schema

**Status:** pending  
**Changes:** ['Add callers array with name, file, line, call_type', 'Add calls array with name, file, line, call_type', 'Add optional call_count field']  

##### ⏳ Add usage tracking fields to class schema

**Status:** pending  
**Changes:** ['Add instantiated_by array', 'Add imported_by array', 'Add property_access tracking (optional for v2.0)']  

##### ⏳ Update schema version to 2.0

**Status:** pending  
**Changes:** ['Update schema_version field', 'Add migration notes', 'Document backward compatibility approach']  

**Blocked by:** task-1-3-1, task-1-3-2

#### ⏳ src/claude_skills/claude_skills/doc_query/doc_query_lib.py

**File:** `src/claude_skills/claude_skills/doc_query/doc_query_lib.py`  
**Status:** pending  
**Estimated:** 8 hours  

**Blocked by:** task-1-3

##### ⏳ Implement get_callers() function

**Status:** pending  
**Changes:** ['Query callers from function data', 'Return formatted caller list', 'Support filtering options']  

##### ⏳ Implement get_callees() function

**Status:** pending  
**Changes:** ['Query calls from function data', 'Return formatted callee list', 'Support filtering options']  

##### ⏳ Implement build_call_graph() function

**Status:** pending  
**Changes:** ['Recursively build call graph', 'Support max depth parameter', 'Return graph structure for visualization']  

**Blocked by:** task-1-4-1, task-1-4-2

#### ⏳ src/claude_skills/claude_skills/doc_query/cli.py

**File:** `src/claude_skills/claude_skills/doc_query/cli.py`  
**Status:** pending  
**Estimated:** 4 hours  

**Blocked by:** task-1-4

##### ⏳ Add 'sdd doc callers' command

**Status:** pending  
**Changes:** ['Create callers CLI command', 'Add --format text|json option', 'Add --docs-path parameter']  

##### ⏳ Add 'sdd doc callees' command

**Status:** pending  
**Changes:** ['Create callees CLI command', 'Add --format text|json option', 'Add --docs-path parameter']  

##### ⏳ Add 'sdd doc call-graph' command

**Status:** pending  
**Changes:** ['Create call-graph CLI command', 'Add --max-depth parameter', 'Add --format dot|json option']  


### Verification (0/5 tasks)

**Blocked by:** phase-1-files  

#### ⏳ Reverse dependencies work correctly

**Status:** pending  
**Type:** manual  

**Command:**
```bash
sdd doc dependencies [module] --reverse
```

**Expected:** Returns all modules that import the target module

#### ⏳ Cross-reference data present in documentation.json

**Status:** pending  
**Type:** manual  

**Expected:** Functions have callers and calls arrays, classes have instantiated_by and imported_by

#### ⏳ New CLI commands work

**Status:** pending  
**Type:** auto  

**Command:**
```bash
sdd doc callers [function] && sdd doc callees [function] && sdd doc call-graph [function]
```

**Expected:** All three commands return valid output

#### ⏳ Schema v2.0 compatibility

**Status:** pending  
**Type:** manual  

**Expected:** Documentation generation produces v2.0 schema with all cross-reference fields

#### ⏳ Cross-reference accuracy on dynamic Python codebase

**Status:** pending  
**Type:** manual  

**Command:**
```bash
Test code-doc skill on codebase with decorators, monkey-patching, and dynamic calls
```

**Expected:** Warnings logged for dynamic patterns; known limitations documented in output; static call detection coverage > 80%


## P1: Core Wrapper Commands (0/18 tasks, 0%)

**Purpose:** Implement 4 high-impact wrapper commands that automate common workflows  
**Risk Level:** medium  
**Estimated Hours:** 110  

**Blocked by:** phase-1  

### File Modifications (0/13 tasks)

#### ⏳ src/claude_skills/claude_skills/doc_query/workflows/trace_entry.py (new file)

**File:** `src/claude_skills/claude_skills/doc_query/workflows/trace_entry.py`  
**Status:** pending  
**Estimated:** 12 hours  

##### ⏳ Implement call chain traversal logic

**Status:** pending  
**Changes:** ['Find entry function', 'Build call chain with max depth', 'Track visited nodes', 'Identify architectural layers']  

##### ⏳ Implement output formatting

**Status:** pending  
**Changes:** ['Tree visualization with layers', 'Complexity highlighting', 'Hot spots identification', 'Dependencies summary']  

**Blocked by:** task-2-1-1

##### ⏳ Add CLI integration

**Status:** pending  
**Changes:** ['Add sdd doc trace-entry command', 'Support --max-depth, --format options']  

**Blocked by:** task-2-1-2

#### ⏳ src/claude_skills/claude_skills/doc_query/workflows/trace_data.py (new file)

**File:** `src/claude_skills/claude_skills/doc_query/workflows/trace_data.py`  
**Status:** pending  
**Estimated:** 12 hours  

##### ⏳ Implement lifecycle operation detection

**Status:** pending  
**Changes:** ['Find class definition', 'Detect creation patterns', 'Detect read/update/delete operations', 'Build data flow graph']  

##### ⏳ Implement output formatting

**Status:** pending  
**Changes:** ['Lifecycle visualization', 'Usage map by layer', 'Property access analysis', 'Mutation hot spots']  

**Blocked by:** task-2-2-1

##### ⏳ Add CLI integration

**Status:** pending  
**Changes:** ['Add sdd doc trace-data command', 'Support --include-properties, --format options']  

**Blocked by:** task-2-2-2

#### ⏳ src/claude_skills/claude_skills/doc_query/workflows/impact_analysis.py (new file)

**File:** `src/claude_skills/claude_skills/doc_query/workflows/impact_analysis.py`  
**Status:** pending  
**Estimated:** 16 hours  

##### ⏳ Implement blast radius calculation

**Status:** pending  
**Changes:** ['Get direct dependents using cross-refs', 'Calculate indirect dependents (2nd degree)', 'Find tests covering entity', 'Calculate risk score']  

##### ⏳ Implement output formatting with recommendations

**Status:** pending  
**Changes:** ['Blast radius visualization', 'Risk matrix display', 'Test coverage report', 'Actionable recommendations', 'Change checklist generation']  

**Blocked by:** task-2-3-1

##### ⏳ Add CLI integration

**Status:** pending  
**Changes:** ['Add sdd doc impact command', 'Support --type, --max-depth, --include-tests, --format options']  

**Blocked by:** task-2-3-2

#### ⏳ src/claude_skills/claude_skills/doc_query/workflows/refactor_candidates.py (new file)

**File:** `src/claude_skills/claude_skills/doc_query/workflows/refactor_candidates.py`  
**Status:** pending  
**Estimated:** 12 hours  

##### ⏳ Implement priority scoring algorithm

**Status:** pending  
**Changes:** ['Find high-complexity functions', 'Get dependent count for each', 'Calculate priority score (complexity × dependents)', 'Categorize risk level']  

##### ⏳ Implement output formatting

**Status:** pending  
**Changes:** ['Ranked table by priority score', 'Priority matrix distribution', 'Recommended action plan', 'Export options (json, csv)']  

**Blocked by:** task-2-4-1

##### ⏳ Add CLI integration

**Status:** pending  
**Changes:** ['Add sdd doc refactor-candidates command', 'Support --threshold, --min-dependents, --format options']  

**Blocked by:** task-2-4-2

#### ⏳ skills/doc-query/SKILL.md

**File:** `skills/doc-query/SKILL.md`  
**Status:** pending  
**Estimated:** 4 hours  

**Blocked by:** task-2-1, task-2-2, task-2-3, task-2-4

##### ⏳ Add documentation for wrapper commands

**Status:** pending  
**Changes:** ['Add examples for trace-entry', 'Add examples for trace-data', 'Add examples for impact', 'Add examples for refactor-candidates', 'Update workflow sections to reference new commands']  


### Verification (0/5 tasks)

**Blocked by:** phase-2-files  

#### ⏳ trace-entry reduces workflow from 6 commands to 1

**Status:** pending  
**Type:** manual  

**Command:**
```bash
sdd doc trace-entry [function]
```

**Expected:** Complete execution flow with layers, complexity, dependencies in single output

#### ⏳ trace-data shows complete lifecycle

**Status:** pending  
**Type:** manual  

**Command:**
```bash
sdd doc trace-data [class]
```

**Expected:** Creation, reading, updating, deletion operations all identified with usage map

#### ⏳ impact analysis uses cross-refs not search

**Status:** pending  
**Type:** manual  

**Command:**
```bash
sdd doc impact [function]
```

**Expected:** Accurate dependent count using callers, not search fallback. Risk assessment included.

#### ⏳ refactor-candidates automates prioritization

**Status:** pending  
**Type:** manual  

**Command:**
```bash
sdd doc refactor-candidates --threshold 10
```

**Expected:** Ranked list with priority scores, risk levels, and action plan

#### ⏳ Output sanitization prevents injection attacks

**Status:** pending  
**Type:** manual  

**Command:**
```bash
Test with crafted function names containing special characters: '; DROP TABLE;', '<script>alert(1)</script>', '../../../etc/passwd', '${evil}'
```

**Expected:** All outputs (dot files, JSON, text) properly escape/sanitize special characters; no command injection possible


## P2: Refinements - Native Filtering (0/6 tasks, 0%)

**Purpose:** Add native pattern filtering to eliminate grep dependencies  
**Risk Level:** low  
**Estimated Hours:** 24  

**Blocked by:** phase-2  

### File Modifications (0/4 tasks)

#### ⏳ src/claude_skills/claude_skills/doc_query/doc_query_lib.py

**File:** `src/claude_skills/claude_skills/doc_query/doc_query_lib.py`  
**Status:** pending  
**Estimated:** 4 hours  

##### ⏳ Add apply_pattern_filter() helper function

**Status:** pending  
**Changes:** ['Create regex pattern matcher', 'Support both string and compiled regex', 'Return filtered results', 'Handle edge cases (invalid regex, etc)']  

#### ⏳ src/claude_skills/claude_skills/doc_query/cli.py

**File:** `src/claude_skills/claude_skills/doc_query/cli.py`  
**Status:** pending  
**Estimated:** 8 hours  

**Blocked by:** task-3-1

##### ⏳ Add --pattern to list-modules command

**Status:** pending  
**Changes:** ['Add --pattern option to CLI', 'Apply filtering before output', 'Preserve JSON output compatibility']  

##### ⏳ Add --pattern to list-classes and list-functions commands

**Status:** pending  
**Changes:** ['Add --pattern option to both commands', 'Apply filtering logic', 'Maintain consistent interface']  

##### ⏳ Add --module-pattern to complexity command

**Status:** pending  
**Changes:** ['Add --module-pattern option', 'Filter by module path before complexity analysis', 'Preserve --json flag functionality']  


### Verification (0/2 tasks)

**Blocked by:** phase-3-files  

#### ⏳ Pattern filtering works on all list commands

**Status:** pending  
**Type:** auto  

**Command:**
```bash
sdd doc list-modules --pattern '(main|index)' && sdd doc list-classes --pattern '.*Service' && sdd doc complexity --module-pattern '*auth*'
```

**Expected:** All commands filter correctly without requiring grep

#### ⏳ JSON output preserved with filtering

**Status:** pending  
**Type:** auto  

**Command:**
```bash
sdd doc list-modules --pattern 'doc' --json | jq .
```

**Expected:** Valid JSON output that can be piped to jq


## P3: Advanced Features (Future) (0/1 tasks, 0%)

**Purpose:** Framework detection, interactive mode, and architecture detection - deferred based on user feedback  
**Risk Level:** medium  
**Estimated Hours:** 56  

**Blocked by:** phase-3  

### Placeholder Tasks (0/1 tasks)

#### ⏳ Future: Architecture detection, interactive mode, framework-aware queries

**File:** `tbd`  
**Status:** pending  
**Estimated:** 56 hours  
