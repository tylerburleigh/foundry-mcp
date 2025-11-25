# Better code-doc Integration for SDD Skills

**Spec ID:** `code-doc-integration-2025-10-24-001`  
**Status:** pending (0/40 tasks, 0%)  
**Estimated Effort:** 16 hours  
**Complexity:** medium  

Integrate code-doc skill into run-tests, sdd-next, and sdd-plan by shifting from optional to recommended workflow with proactive generation

## run-tests Skill Integration (0/9 tasks, 0%)

**Purpose:** Add code-doc integration to run-tests skill (currently has NO integration)  
**Risk Level:** low  
**Estimated Hours:** 4  


### File Modifications (0/5 tasks)

#### ⏳ skills/run-tests/SKILL.md

**File:** `skills/run-tests/SKILL.md`  
**Status:** pending  
**Estimated:** 3 hours  
**Changes:** Add code-doc integration sections  

##### ⏳ Add Phase 2.4: Codebase Context Analysis section

**Status:** pending  

##### ⏳ Enhance Phase 3: Add Step 1a for documentation usage

**Status:** pending  

**Depends on:** task-1-1-1

##### ⏳ Update Best Practices section

**Status:** pending  

##### ⏳ Update External Tool Consultation section

**Status:** pending  

##### ⏳ Add command reference table

**Status:** pending  

**Depends on:** task-1-1-1, task-1-1-2


### Verification (0/4 tasks)

**Blocked by:** phase-1-files  

#### ⏳ Verify all sections added at correct locations

**Status:** pending  
**Type:** manual  

**Expected:** All 4 new sections appear at specified line numbers with proper markdown formatting

#### ⏳ Verify command examples are accurate

**Status:** pending  
**Type:** manual  

**Expected:** All sdd doc commands use correct syntax and realistic examples

#### ⏳ Verify graceful degradation paths are clear

**Status:** pending  
**Type:** manual  

**Expected:** Documentation explains what to do when code-doc unavailable or user declines

#### ⏳ Verify markdown formatting is valid

**Status:** pending  
**Type:** auto  

**Command:**
```bash
markdownlint skills/run-tests/SKILL.md
```

**Expected:** No markdown linting errors


## sdd-next Skill Enhancement (0/10 tasks, 0%)

**Purpose:** Upgrade from optional to recommended primary workflow with proactive generation  
**Risk Level:** medium  
**Estimated Hours:** 4  

**Blocked by:** phase-1  

### File Modifications (0/6 tasks)

#### ⏳ skills/sdd-next/SKILL.md

**File:** `skills/sdd-next/SKILL.md`  
**Status:** pending  
**Estimated:** 3.5 hours  
**Changes:** Elevate code-doc to primary workflow method  

##### ⏳ Enhance Phase 1.3: Change to documentation-first workflow

**Status:** pending  

##### ⏳ Enhance Phase 3.1: Make code-doc primary method

**Status:** pending  

##### ⏳ Add Proactive Documentation Generation section

**Status:** pending  

##### ⏳ Update Quick Start section

**Status:** pending  

##### ⏳ Add command reference table

**Status:** pending  

**Depends on:** task-2-1-1, task-2-1-2

##### ⏳ Add workflow example showing doc context usage

**Status:** pending  

**Depends on:** task-2-1-1


### Verification (0/4 tasks)

**Blocked by:** phase-2-files  

#### ⏳ Verify language changed from optional to recommended

**Status:** pending  
**Type:** manual  

**Expected:** All instances of 'optional' changed to 'recommended' or 'primary workflow'

#### ⏳ Verify proactive generation workflow is clear

**Status:** pending  
**Type:** manual  

**Expected:** Auto-detection pattern and user prompts are clear and actionable

#### ⏳ Verify workflow ordering is correct

**Status:** pending  
**Type:** manual  

**Expected:** Documentation-first comes before manual fallback in all sections

#### ⏳ Verify markdown formatting is valid

**Status:** pending  
**Type:** auto  

**Command:**
```bash
markdownlint skills/sdd-next/SKILL.md
```

**Expected:** No markdown linting errors


## sdd-plan Skill Enhancement (0/9 tasks, 0%)

**Purpose:** Polish existing integration, add proactive generation, remove optional language  
**Risk Level:** low  
**Estimated Hours:** 3  

**Blocked by:** phase-2  

### File Modifications (0/5 tasks)

#### ⏳ skills/sdd-plan/SKILL.md

**File:** `skills/sdd-plan/SKILL.md`  
**Status:** pending  
**Estimated:** 2.5 hours  
**Changes:** Polish existing code-doc integration and add proactive generation  

##### ⏳ Enhance Section 1.2.1: Add proactive generation workflow

**Status:** pending  

##### ⏳ Update Phase 1.2 header with documentation-first reference

**Status:** pending  

##### ⏳ Add Best Practices for Codebase Analysis subsection

**Status:** pending  

##### ⏳ Update Core Philosophy section

**Status:** pending  

##### ⏳ Add command reference table

**Status:** pending  

**Depends on:** task-3-1-1


### Verification (0/4 tasks)

**Blocked by:** phase-3-files  

#### ⏳ Verify all optional language removed

**Status:** pending  
**Type:** manual  

**Expected:** No instances of 'optional' or 'if available' remain in code-doc sections

#### ⏳ Verify proactive generation workflow is comprehensive

**Status:** pending  
**Type:** manual  

**Expected:** Auto-detection, user prompt, and acceptance/decline paths are all documented

#### ⏳ Verify consistency with run-tests and sdd-next patterns

**Status:** pending  
**Type:** manual  

**Expected:** Language, structure, and prompts match patterns from phases 1-2

#### ⏳ Verify markdown formatting is valid

**Status:** pending  
**Type:** auto  

**Command:**
```bash
markdownlint skills/sdd-plan/SKILL.md
```

**Expected:** No markdown linting errors


## Cross-Cutting Documentation (0/7 tasks, 0%)

**Purpose:** Add standardized patterns, examples, and reference materials across all skills  
**Risk Level:** low  
**Estimated Hours:** 3  

**Blocked by:** phase-3  

### File Modifications (0/3 tasks)

#### ⏳ Add before/after workflow examples to all three skills

**File:** `skills/run-tests/SKILL.md, skills/sdd-next/SKILL.md, skills/sdd-plan/SKILL.md`  
**Status:** pending  
**Estimated:** 1 hours  

#### ⏳ Add failure mode handling documentation to all three skills

**File:** `skills/run-tests/SKILL.md, skills/sdd-next/SKILL.md, skills/sdd-plan/SKILL.md`  
**Status:** pending  
**Estimated:** 1 hours  

#### ⏳ Standardize wording transformations across all three skills

**File:** `skills/run-tests/SKILL.md, skills/sdd-next/SKILL.md, skills/sdd-plan/SKILL.md`  
**Status:** pending  
**Estimated:** 1 hours  


### Verification (0/4 tasks)

**Blocked by:** phase-4-files  

#### ⏳ Verify workflow examples are accurate and helpful

**Status:** pending  
**Type:** manual  

**Expected:** Before/after examples clearly demonstrate value of code-doc integration

#### ⏳ Verify failure mode documentation is comprehensive

**Status:** pending  
**Type:** manual  

**Expected:** All three failure scenarios documented with clear resolution paths

#### ⏳ Verify wording is consistent across all three skills

**Status:** pending  
**Type:** manual  

**Expected:** Same terminology and phrasing used in equivalent sections across skills

#### ⏳ Verify cross-references between skills are accurate

**Status:** pending  
**Type:** manual  

**Expected:** All Skill() references and cross-skill links are correct


## Validation and Quality Assurance (0/5 tasks, 0%)

**Purpose:** Verify all changes maintain consistency, accuracy, and graceful degradation  
**Risk Level:** low  
**Estimated Hours:** 2  

**Blocked by:** phase-4  

### Verification (0/5 tasks)

#### ⏳ Verify all line number references are accurate

**Status:** pending  
**Type:** manual  

**Expected:** All location references match actual line numbers in modified files

#### ⏳ Test proactive generation prompts for clarity

**Status:** pending  
**Type:** manual  

**Expected:** User prompts are clear, actionable, and include value proposition

#### ⏳ Confirm graceful degradation paths work

**Status:** pending  
**Type:** manual  

**Expected:** All three skills have clear fallback workflows when code-doc unavailable

#### ⏳ Validate command examples are correct

**Status:** pending  
**Type:** manual  

**Expected:** All sdd doc commands use correct syntax and realistic scenarios

#### ⏳ Final markdown linting check for all three skills

**Status:** pending  
**Type:** auto  

**Command:**
```bash
markdownlint skills/run-tests/SKILL.md skills/sdd-next/SKILL.md skills/sdd-plan/SKILL.md
```

**Expected:** No markdown linting errors in any skill file
