# Refactor SDD Skills to Use Claude Code Subagents

**Spec ID:** `subagent-refactoring-2025-10-28-001`  
**Status:** pending (0/45 tasks, 0%)  
**Estimated Effort:** 24 hours  
**Complexity:** high  

Refactor SDD skills to use Claude Code Task tool subagents for action-oriented operations with result reporting

## Objectives

- Create subagent definitions for sdd-update, sdd-plan-review, sdd-validate, run-tests, and code-doc
- Update sdd-plan and sdd-next to invoke these skills as subagents by default
- Ensure subagents perform work/research and report results back
- Maintain natural language output format
- Keep doc-query as direct import for frequent usage

## Create Subagent Definitions (0/16 tasks, 0%)

**Purpose:** Create minimal SKILL.md subagent definitions that invoke actual skills and report results  
**Risk Level:** low  
**Estimated Hours:** 6  


### File Modifications (0/10 tasks)

#### ⏳ Create subagent definition for sdd-update

**File:** `skills/subagents/sdd-update/SKILL.md`  
**Status:** pending  

##### ⏳ Write SKILL.md with subagent instructions

**Status:** pending  

##### ⏳ Add reporting format instructions

**Status:** pending  

#### ⏳ Create subagent definition for sdd-plan-review

**File:** `skills/subagents/sdd-plan-review/SKILL.md`  
**Status:** pending  

##### ⏳ Write SKILL.md with review invocation

**Status:** pending  

##### ⏳ Define review result reporting format

**Status:** pending  

#### ⏳ Create subagent definition for sdd-validate

**File:** `skills/subagents/sdd-validate/SKILL.md`  
**Status:** pending  

##### ⏳ Write SKILL.md for validation workflow

**Status:** pending  

##### ⏳ Define validation result reporting

**Status:** pending  

#### ⏳ Create subagent definition for run-tests

**File:** `skills/subagents/run-tests/SKILL.md`  
**Status:** pending  

##### ⏳ Write SKILL.md for test execution

**Status:** pending  

##### ⏳ Define test result reporting format

**Status:** pending  

#### ⏳ Create subagent definition for code-doc

**File:** `skills/subagents/code-doc/SKILL.md`  
**Status:** pending  

##### ⏳ Write SKILL.md for documentation generation

**Status:** pending  

##### ⏳ Define documentation result reporting

**Status:** pending  


### Verification (0/6 tasks)

**Blocked by:** phase-1-files  

#### ⏳ Subagent SKILL.md files are readable and well-formatted

**Status:** pending  
**Type:** manual  

**Expected:** All 5 SKILL.md files exist and follow consistent format with clear instructions

#### ⏳ Each subagent includes when-to-use guidance

**Status:** pending  
**Type:** manual  

**Expected:** Each SKILL.md clearly states when the subagent should be invoked

#### ⏳ Each subagent includes CLI invocation examples

**Status:** pending  
**Type:** manual  

**Expected:** SKILL.md files show concrete examples of invoking sdd CLI commands

#### ⏳ Each subagent includes result reporting format

**Status:** pending  
**Type:** manual  

**Expected:** Each SKILL.md specifies what to report back in natural language

#### ⏳ Subagent instructions emphasize action and reporting

**Status:** pending  
**Type:** manual  

**Expected:** Instructions clearly state: perform work/research, then report results

#### ⏳ No JSON output flags required in subagent definitions

**Status:** pending  
**Type:** manual  

**Expected:** Subagents use natural language output, no --json flags in examples


## Update sdd-plan Workflow (0/9 tasks, 0%)

**Purpose:** Modify sdd-plan to invoke sdd-plan-review, code-doc, and sdd-validate as subagents by default  
**Risk Level:** medium  
**Estimated Hours:** 6  

**Blocked by:** phase-1  

### File Modifications (0/6 tasks)

#### ⏳ Update sdd-plan SKILL.md to reference subagents

**File:** `skills/sdd-plan/SKILL.md`  
**Status:** pending  

##### ⏳ Add section on invoking sdd-validate subagent after spec creation

**Status:** pending  

##### ⏳ Add section on invoking sdd-plan-review subagent for spec review

**Status:** pending  

##### ⏳ Add section on invoking code-doc subagent for codebase analysis

**Status:** pending  

#### ⏳ Update sdd-plan workflow examples

**File:** `skills/sdd-plan/SKILL.md`  
**Status:** pending  

**Depends on:** task-2-1

##### ⏳ Replace validation command examples with subagent calls

**Status:** pending  

##### ⏳ Add example workflow showing subagent orchestration

**Status:** pending  

#### ⏳ Remove direct CLI command references from sdd-plan

**File:** `skills/sdd-plan/SKILL.md`  
**Status:** pending  

**Depends on:** task-2-1, task-2-2


### Verification (0/3 tasks)

**Blocked by:** phase-2-files  

#### ⏳ sdd-plan SKILL.md references all three subagents

**Status:** pending  
**Type:** manual  

**Expected:** SKILL.md mentions sdd-validate, sdd-plan-review, and code-doc subagents

#### ⏳ No direct CLI validation/review commands remain

**Status:** pending  
**Type:** auto  

**Command:**
```bash
grep -n 'sdd validate\|sdd review\|sdd doc' skills/sdd-plan/SKILL.md | grep -v 'subagent\|Task'
```

**Expected:** No matches found (all references should be via subagent)

#### ⏳ Workflow examples show Task tool usage

**Status:** pending  
**Type:** manual  

**Expected:** Examples demonstrate Task(subagent_type=...) pattern


## Update sdd-next Workflow (0/9 tasks, 0%)

**Purpose:** Modify sdd-next to invoke sdd-validate, run-tests, and code-doc as subagents by default  
**Risk Level:** medium  
**Estimated Hours:** 6  

**Blocked by:** phase-1  

### File Modifications (0/6 tasks)

#### ⏳ Update sdd-next SKILL.md to reference subagents

**File:** `skills/sdd-next/SKILL.md`  
**Status:** pending  

##### ⏳ Add section on invoking sdd-validate subagent before task preparation

**Status:** pending  

##### ⏳ Add section on invoking run-tests subagent for verification tasks

**Status:** pending  

##### ⏳ Add section on invoking code-doc subagent for context gathering

**Status:** pending  

#### ⏳ Update sdd-next workflow examples

**File:** `skills/sdd-next/SKILL.md`  
**Status:** pending  

**Depends on:** task-3-1

##### ⏳ Replace test execution examples with subagent calls

**Status:** pending  

##### ⏳ Add example of validation subagent in prepare-task workflow

**Status:** pending  

#### ⏳ Remove direct CLI command references from sdd-next

**File:** `skills/sdd-next/SKILL.md`  
**Status:** pending  

**Depends on:** task-3-1, task-3-2


### Verification (0/3 tasks)

**Blocked by:** phase-3-files  

#### ⏳ sdd-next SKILL.md references all three subagents

**Status:** pending  
**Type:** manual  

**Expected:** SKILL.md mentions sdd-validate, run-tests, and code-doc subagents

#### ⏳ No direct CLI test/validation commands remain

**Status:** pending  
**Type:** auto  

**Command:**
```bash
grep -n 'sdd validate\|run-tests:\|pytest' skills/sdd-next/SKILL.md | grep -v 'subagent\|Task'
```

**Expected:** No matches found (all references should be via subagent)

#### ⏳ Workflow examples show Task tool usage

**Status:** pending  
**Type:** manual  

**Expected:** Examples demonstrate Task(subagent_type=...) pattern for validation, testing, doc generation


## Update Cross-References and Documentation (0/11 tasks, 0%)

**Purpose:** Update other skill references and main documentation to reflect subagent architecture  
**Risk Level:** low  
**Estimated Hours:** 6  

**Blocked by:** phase-2, phase-3  

### File Modifications (0/8 tasks)

#### ⏳ Update sdd-update SKILL.md cross-references

**File:** `skills/sdd-update/SKILL.md`  
**Status:** pending  

##### ⏳ Replace validation CLI references with subagent

**Status:** pending  

#### ⏳ Update main README or toolkit docs

**File:** `README.md`  
**Status:** pending  

##### ⏳ Add section explaining subagent architecture

**Status:** pending  

##### ⏳ List which skills are subagents vs direct

**Status:** pending  

#### ⏳ Create subagent architecture diagram

**File:** `docs/architecture/subagents.md`  
**Status:** pending  

##### ⏳ Draw workflow showing sdd-plan/sdd-next invoking subagents

**Status:** pending  

#### ⏳ Update plugin.json subagent definitions

**File:** `plugin.json`  
**Status:** pending  

##### ⏳ Add 5 new subagent skill entries

**Status:** pending  

##### ⏳ Mark subagent skills as type 'subagent'

**Status:** pending  

#### ⏳ Update other skill cross-references

**File:** `multiple`  
**Status:** pending  

##### ⏳ Search for validation/review/test references

**Status:** pending  

##### ⏳ Update found references to use subagents

**Status:** pending  


### Verification (0/3 tasks)

**Blocked by:** phase-4-files  

#### ⏳ All cross-references updated to subagents

**Status:** pending  
**Type:** auto  

**Command:**
```bash
grep -r 'sdd validate\|sdd review\|sdd-review\|run-tests:' skills/*/SKILL.md | grep -v subagent | grep -v '# Examples' | wc -l
```

**Expected:** 0 (no direct CLI references outside of examples/comments)

#### ⏳ Documentation clearly explains subagent architecture

**Status:** pending  
**Type:** manual  

**Expected:** README or docs explain which skills are subagents and why

#### ⏳ plugin.json includes all subagent skills

**Status:** pending  
**Type:** auto  

**Command:**
```bash
grep -c 'subagent-' plugin.json
```

**Expected:** 5 (one entry for each subagent skill)
