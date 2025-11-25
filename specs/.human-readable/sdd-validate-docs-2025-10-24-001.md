# sdd-validate Documentation Improvements

**Spec ID:** `sdd-validate-docs-2025-10-24-001`  
**Status:** pending (0/11 tasks, 0%)  
**Estimated Effort:** 4 hours  
**Complexity:** low  

Improve SKILL.md to clarify exit codes, iterative workflow, auto-fix limitations, and add troubleshooting guidance

## Documentation Updates (0/7 tasks, 0%)

**Purpose:** Update SKILL.md with clarifications, workflow improvements, and troubleshooting guidance  
**Risk Level:** low  
**Estimated Hours:** 2  


### File Modifications (0/5 tasks)

#### ⏳ skills/sdd-validate/SKILL.md

**File:** `skills/sdd-validate/SKILL.md`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add clarifications, workflow guidance, troubleshooting, and decision trees  

##### ⏳ Enhance exit code explanation section

**Status:** pending  

**Details:** Update lines 82-86 and 102-105 to emphasize exit code 2 is 'validation found errors' not 'command failed'. Add note for AI assistants that this is expected behavior.

##### ⏳ Add iterative fixing workflow section

**Status:** pending  

**Details:** Add new section after line 220 explaining: (1) may need multiple fix passes, (2) some fixes enable detection of new issues, (3) when to stop iterating, (4) fix success doesn't mean all issues fixed

**Blocked by:** task-1-1-1

##### ⏳ Update auto-fix documentation with limitations

**Status:** pending  

**Details:** Update lines 120-133 to add note that 'auto-fixable' doesn't guarantee fix will be applied - some issues require context. Explain exit 0 from fix means 'applied what it could' not 'fixed everything'

**Blocked by:** task-1-1-1

##### ⏳ Add troubleshooting common issues section

**Status:** pending  

**Details:** Add new section after validation workflow covering: (1) 'Auto-fix succeeded but validation still shows errors', (2) 'How many times should I run fix?', (3) 'When to give up on auto-fix and fix manually?', (4) Real scenario from our experience

**Blocked by:** task-1-1-2, task-1-1-3

##### ⏳ Add decision tree for stopping auto-fix

**Status:** pending  

**Details:** Add flowchart/decision tree: Run validate → Errors? → Run fix → Re-validate → Same errors? → Check if auto-fixable list changed → No change = needs manual fix → Yes change = run fix again

**Blocked by:** task-1-1-4


### Verification (0/2 tasks)

**Blocked by:** phase-1-files  

#### ⏳ Documentation structure is valid markdown

**Status:** pending  
**Type:** auto  

**Command:**
```bash
markdownlint skills/sdd-validate/SKILL.md || echo 'No markdown linter installed - manual review required'
```

**Expected:** Markdown is well-formed with proper heading hierarchy

#### ⏳ All sections are clear and actionable

**Status:** pending  
**Type:** manual  

**Command:**
```bash
# Manual review required
```

**Expected:** Read through documentation ensuring: (1) exit code explanation is unambiguous, (2) workflow steps are sequential and clear, (3) troubleshooting covers our scenario


## Examples and Verification (0/4 tasks, 0%)

**Purpose:** Add practical examples and validate documentation clarity  
**Risk Level:** low  
**Estimated Hours:** 2  

**Blocked by:** phase-1  

### File Modifications (0/3 tasks)

#### ⏳ skills/sdd-validate/SKILL.md

**File:** `skills/sdd-validate/SKILL.md`  
**Status:** pending  
**Estimated:** 1.5 hours  
**Changes:** Add real-world examples, code snippets, and AI assistant guidelines  

**Blocked by:** phase-1

##### ⏳ Add real-world workflow examples

**Status:** pending  

**Details:** Add example based on our experience: 88 errors → fix → 4 errors → fix → still 4 errors → requires manual intervention. Show complete terminal output and interpretation.

##### ⏳ Add code snippets with expected output

**Status:** pending  

**Details:** Show actual sdd validate/fix command output with exit codes, error counts, and auto-fix status. Include before/after examples showing successful partial fixes.

**Blocked by:** task-2-1-1

##### ⏳ Add AI assistant guidelines

**Status:** pending  

**Details:** Add callout box or section explaining: (1) Don't interpret exit 2 as failure, (2) Always re-validate after fix, (3) Check error count progression, (4) When to report issues vs continue fixing

**Blocked by:** task-2-1-2


### Verification (0/1 tasks)

**Blocked by:** phase-2-files  

#### ⏳ Examples are accurate and match tool behavior

**Status:** pending  
**Type:** manual  

**Command:**
```bash
# Manual verification required
```

**Expected:** Run the example commands from documentation and verify: (1) exit codes match, (2) output format matches, (3) workflow produces described results, (4) AI assistant guidelines prevent our original confusion
