# Better code-doc Integration for SDD Skills (Implementation + Documentation)

**Spec ID:** `code-doc-integration-2025-10-24-002`  
**Status:** pending (0/60 tasks, 0%)  
**Estimated Effort:** 32 hours  
**Complexity:** high  

Integrate code-doc skill into run-tests, sdd-next, and sdd-plan with both Python implementation and documentation updates. Includes shared utility, proactive generation, and comprehensive testing.

## Foundation: Shared Utilities and Contract (0/8 tasks, 0%)

**Purpose:** Create shared utility for proactive doc checking and define sdd doc stats contract  
**Risk Level:** medium  
**Estimated Hours:** 4  


### File Modifications (0/4 tasks)

#### ⏳ src/sdd_common/doc_integration.py

**File:** `src/sdd_common/doc_integration.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Create new shared utility module for doc-checking logic  

##### ⏳ Implement check_doc_availability() function

**Status:** pending  

##### ⏳ Implement prompt_for_generation() function

**Status:** pending  

**Depends on:** task-1-1-1

#### ⏳ docs/contracts/sdd-doc-stats-contract.md

**File:** `docs/contracts/sdd-doc-stats-contract.md`  
**Status:** pending  
**Estimated:** 1 hours  
**Changes:** Create new contract document defining sdd doc stats behavior  

##### ⏳ Define sdd doc stats exit codes

**Status:** pending  

##### ⏳ Define sdd doc stats output format

**Status:** pending  


### Verification (0/4 tasks)

**Blocked by:** phase-1-files  

#### ⏳ Unit test check_doc_availability()

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/sdd_common/test_doc_integration.py::test_check_doc_availability
```

**Expected:** Tests pass for all DocStatus scenarios

#### ⏳ Unit test prompt_for_generation()

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/sdd_common/test_doc_integration.py::test_prompt_for_generation
```

**Expected:** Tests pass for Y/n/timeout scenarios

#### ⏳ Verify contract document completeness

**Status:** pending  
**Type:** manual  

**Expected:** Contract defines all exit codes, output formats, and error conditions

#### ⏳ Verify shared utility can be imported

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -c 'from sdd_common.doc_integration import check_doc_availability'
```

**Expected:** Import succeeds without errors


## Implementation: run-tests Skill (0/9 tasks, 0%)

**Purpose:** Implement proactive doc checking in run-tests Python code  
**Risk Level:** medium  
**Estimated Hours:** 4  

**Blocked by:** phase-1  

### File Modifications (0/5 tasks)

#### ⏳ skills/run-tests/run_tests.py

**File:** `skills/run-tests/run_tests.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add proactive doc checking to test failure analysis workflow  

##### ⏳ Import shared doc_integration utility

**Status:** pending  

##### ⏳ Add proactive check in test failure analysis

**Status:** pending  

**Depends on:** task-2-1-1

##### ⏳ Handle user response and graceful degradation

**Status:** pending  

**Depends on:** task-2-1-2

#### ⏳ tests/run_tests/test_doc_integration.py

**File:** `tests/run_tests/test_doc_integration.py`  
**Status:** pending  
**Estimated:** 1.5 hours  
**Changes:** Create new test file for doc integration in run-tests  

**Depends on:** task-2-1

##### ⏳ Test proactive check is called

**Status:** pending  

##### ⏳ Test graceful degradation

**Status:** pending  


### Verification (0/4 tasks)

**Blocked by:** phase-2-files  

#### ⏳ Run run-tests unit tests

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/run_tests/test_doc_integration.py -v
```

**Expected:** All tests pass

#### ⏳ Integration test: docs available path

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_run_tests_doc_integration.py::test_docs_available
```

**Expected:** Workflow uses docs when available

#### ⏳ Integration test: docs missing, user accepts

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_run_tests_doc_integration.py::test_docs_missing_accept
```

**Expected:** code-doc skill is called, then workflow continues

#### ⏳ Integration test: docs missing, user declines

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_run_tests_doc_integration.py::test_docs_missing_decline
```

**Expected:** Graceful degradation: workflow continues without docs


## Implementation: sdd-next Skill (0/9 tasks, 0%)

**Purpose:** Implement proactive doc checking in sdd-next Python code  
**Risk Level:** medium  
**Estimated Hours:** 4  

**Blocked by:** phase-1  

### File Modifications (0/5 tasks)

#### ⏳ skills/sdd-next/sdd_next.py

**File:** `skills/sdd-next/sdd_next.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add proactive doc checking to task preparation workflow  

##### ⏳ Import shared doc_integration utility

**Status:** pending  

##### ⏳ Add proactive check in task preparation

**Status:** pending  

**Depends on:** task-3-1-1

##### ⏳ Handle user response and graceful degradation

**Status:** pending  

**Depends on:** task-3-1-2

#### ⏳ tests/sdd_next/test_doc_integration.py

**File:** `tests/sdd_next/test_doc_integration.py`  
**Status:** pending  
**Estimated:** 1.5 hours  
**Changes:** Create new test file for doc integration in sdd-next  

**Depends on:** task-3-1

##### ⏳ Test proactive check is called

**Status:** pending  

##### ⏳ Test graceful degradation

**Status:** pending  


### Verification (0/4 tasks)

**Blocked by:** phase-3-files  

#### ⏳ Run sdd-next unit tests

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/sdd_next/test_doc_integration.py -v
```

**Expected:** All tests pass

#### ⏳ Integration test: docs available path

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_sdd_next_doc_integration.py::test_docs_available
```

**Expected:** Workflow uses sdd doc context when available

#### ⏳ Integration test: docs missing, user accepts

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_sdd_next_doc_integration.py::test_docs_missing_accept
```

**Expected:** code-doc skill is called, then context gathering proceeds

#### ⏳ Integration test: docs missing, user declines

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_sdd_next_doc_integration.py::test_docs_missing_decline
```

**Expected:** Graceful degradation: manual exploration path used


## Implementation: sdd-plan Skill (0/9 tasks, 0%)

**Purpose:** Implement proactive doc checking in sdd-plan Python code  
**Risk Level:** medium  
**Estimated Hours:** 4  

**Blocked by:** phase-1  

### File Modifications (0/5 tasks)

#### ⏳ skills/sdd-plan/sdd_plan.py

**File:** `skills/sdd-plan/sdd_plan.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add proactive doc checking to planning workflow  

##### ⏳ Import shared doc_integration utility

**Status:** pending  

##### ⏳ Add proactive check in codebase analysis

**Status:** pending  

**Depends on:** task-4-1-1

##### ⏳ Handle user response and graceful degradation

**Status:** pending  

**Depends on:** task-4-1-2

#### ⏳ tests/sdd_plan/test_doc_integration.py

**File:** `tests/sdd_plan/test_doc_integration.py`  
**Status:** pending  
**Estimated:** 1.5 hours  
**Changes:** Create new test file for doc integration in sdd-plan  

**Depends on:** task-4-1

##### ⏳ Test proactive check is called

**Status:** pending  

##### ⏳ Test graceful degradation

**Status:** pending  


### Verification (0/4 tasks)

**Blocked by:** phase-4-files  

#### ⏳ Run sdd-plan unit tests

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/sdd_plan/test_doc_integration.py -v
```

**Expected:** All tests pass

#### ⏳ Integration test: docs available path

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_sdd_plan_doc_integration.py::test_docs_available
```

**Expected:** Workflow uses doc queries when available

#### ⏳ Integration test: docs missing, user accepts

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_sdd_plan_doc_integration.py::test_docs_missing_accept
```

**Expected:** code-doc skill is called, then planning proceeds

#### ⏳ Integration test: docs missing, user declines

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_sdd_plan_doc_integration.py::test_docs_missing_decline
```

**Expected:** Graceful degradation: manual analysis path used


## Documentation: Update SKILL.md Files (0/10 tasks, 0%)

**Purpose:** Update skill documentation to reflect new proactive workflow  
**Risk Level:** low  
**Estimated Hours:** 6  

**Blocked by:** phase-2, phase-3, phase-4  

### File Modifications (0/6 tasks)

#### ⏳ skills/run-tests/SKILL.md

**File:** `skills/run-tests/SKILL.md`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add documentation about proactive doc checking  

##### ⏳ Add section explaining proactive doc checking

**Status:** pending  

##### ⏳ Add command reference table

**Status:** pending  

**Depends on:** task-5-1-1

#### ⏳ skills/sdd-next/SKILL.md

**File:** `skills/sdd-next/SKILL.md`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add documentation about proactive doc checking  

##### ⏳ Add section explaining proactive doc checking

**Status:** pending  

##### ⏳ Add command reference table

**Status:** pending  

**Depends on:** task-5-2-1

#### ⏳ skills/sdd-plan/SKILL.md

**File:** `skills/sdd-plan/SKILL.md`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add documentation about proactive doc checking  

##### ⏳ Add section explaining proactive doc checking

**Status:** pending  

##### ⏳ Add command reference table

**Status:** pending  

**Depends on:** task-5-3-1


### Verification (0/4 tasks)

**Blocked by:** phase-5-files  

#### ⏳ Verify markdown formatting is valid

**Status:** pending  
**Type:** auto  

**Command:**
```bash
markdownlint skills/run-tests/SKILL.md skills/sdd-next/SKILL.md skills/sdd-plan/SKILL.md
```

**Expected:** No markdown linting errors

#### ⏳ Verify command examples are accurate

**Status:** pending  
**Type:** manual  

**Expected:** All sdd doc commands use correct syntax and realistic examples

#### ⏳ Verify sections added at correct locations

**Status:** pending  
**Type:** manual  

**Expected:** All new sections appear after specified section identifiers with proper hierarchy

#### ⏳ Verify graceful degradation is documented

**Status:** pending  
**Type:** manual  

**Expected:** Documentation clearly explains what happens when docs unavailable or user declines


## End-to-End Testing (0/7 tasks, 0%)

**Purpose:** Test complete workflows across all three skills  
**Risk Level:** medium  
**Estimated Hours:** 4  

**Blocked by:** phase-2, phase-3, phase-4, phase-5  

### File Modifications (0/3 tasks)

#### ⏳ tests/e2e/test_doc_integration_workflows.py

**File:** `tests/e2e/test_doc_integration_workflows.py`  
**Status:** pending  
**Estimated:** 3 hours  
**Changes:** Create end-to-end workflow tests  

##### ⏳ Test run-tests complete workflow

**Status:** pending  

##### ⏳ Test sdd-next complete workflow

**Status:** pending  

##### ⏳ Test sdd-plan complete workflow

**Status:** pending  


### Verification (0/4 tasks)

**Blocked by:** phase-6-files  

#### ⏳ Run all E2E tests

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/e2e/test_doc_integration_workflows.py -v
```

**Expected:** All E2E tests pass

#### ⏳ Manual smoke test: run-tests with missing docs

**Status:** pending  
**Type:** manual  

**Expected:** Skill prompts for doc generation, accepts input, calls code-doc skill

#### ⏳ Manual smoke test: sdd-next with available docs

**Status:** pending  
**Type:** manual  

**Expected:** Skill uses docs automatically, shows value message, context gathering succeeds

#### ⏳ Manual smoke test: sdd-plan with user decline

**Status:** pending  
**Type:** manual  

**Expected:** Skill gracefully degrades to manual analysis, no errors, planning succeeds


## Final Validation and Documentation (0/4 tasks, 0%)

**Purpose:** Final checks and create integration guide  
**Risk Level:** low  
**Estimated Hours:** 2  

**Blocked by:** phase-6  

### Verification (0/4 tasks)

#### ⏳ Run full test suite

**Status:** pending  
**Type:** auto  

**Command:**
```bash
pytest tests/ -v --cov=src/sdd_common --cov=skills/run-tests --cov=skills/sdd-next --cov=skills/sdd-plan
```

**Expected:** All tests pass, coverage > 80%

#### ⏳ Verify no regressions in existing workflows

**Status:** pending  
**Type:** manual  

**Expected:** All three skills work normally when docs are already available (no prompts shown)

#### ⏳ Verify sdd doc stats contract compliance

**Status:** pending  
**Type:** manual  

**Expected:** Integration respects all exit codes and output formats defined in contract

#### ⏳ Create integration guide document

**Status:** pending  
**Type:** manual  

**Expected:** docs/code-doc-integration-guide.md created with usage examples and troubleshooting
