# AI Consultation Infrastructure Refactoring

**Spec ID:** `ai-consultation-refactor-2025-11-05-001`  
**Status:** completed (51/51 tasks, 100%)  
**Estimated Effort:** 35 hours  
**Complexity:** high  

Extract and consolidate duplicated AI consultation code from run-tests, sdd-plan-review, and code-doc into shared utilities, eliminating ~850 lines of duplication

## Objectives

- Eliminate ~850 lines of code duplication across 3 existing skills
- Create shared utilities in common/ai_tools.py for reusable consultation infrastructure
- Extend common/ai_config.py for multi-agent configuration support
- Maintain backward compatibility and existing functionality for all skills
- Provide proven infrastructure for future skills requiring AI consultation

## Shared Utilities Creation (12/12 tasks, 100%)

**Purpose:** Create foundational common/ai_tools.py module with all shared consultation functionality  
**Risk Level:** low  
**Estimated Hours:** 8  


### File Modifications (10/10 tasks)

#### ✅ Analyze existing consultation implementations

**Status:** completed  
**Estimated:** 2 hours  

##### ✅ Document tool availability checking patterns

**Status:** completed  

##### ✅ Document command building and execution patterns

**Status:** completed  

#### ✅ Design ToolResponse dataclass and core interfaces

**Status:** completed  
**Estimated:** 1 hours  

**Blocked by:** task-1-1

#### ✅ src/claude_skills/claude_skills/common/ai_tools.py

**File:** `src/claude_skills/claude_skills/common/ai_tools.py`  
**Status:** completed  
**Estimated:** 5 hours  

**Blocked by:** task-1-2

##### ✅ Implement ToolResponse dataclass

**Status:** completed  

##### ✅ Implement check_tool_available() and detect_available_tools()

**Status:** completed  

##### ✅ Implement build_tool_command()

**Status:** completed  

##### ✅ Implement execute_tool() with timeout handling

**Status:** completed  

##### ✅ Implement execute_tools_parallel()

**Status:** completed  

##### ✅ Add comprehensive docstrings and type hints

**Status:** completed  

#### ✅ tests/test_ai_tools.py

**File:** `tests/test_ai_tools.py`  
**Status:** completed  
**Estimated:** 0.5 hours  

**Blocked by:** task-1-3


### Verification (2/2 tasks)

**Blocked by:** phase-1-files  

#### ✅ Unit tests pass for ai_tools module

**Status:** completed  
**Type:** auto  

**Command:**
```bash
pytest tests/test_ai_tools.py -v
```

**Expected:** All tests pass

#### ✅ Module imports successfully

**Status:** completed  
**Type:** auto  

**Command:**
```bash
python -c 'from claude_skills.common.ai_tools import *'
```

**Expected:** No import errors


## Config Extension (4/4 tasks, 100%)

**Purpose:** Extend existing common/ai_config.py to support multi-agent pairs and routing configurations  
**Risk Level:** low  
**Estimated Hours:** 2  

**Blocked by:** phase-1  

### File Modifications (2/2 tasks)

#### ✅ src/claude_skills/claude_skills/common/ai_config.py

**File:** `src/claude_skills/claude_skills/common/ai_config.py`  
**Status:** completed  
**Estimated:** 1.5 hours  

##### ✅ Implement get_multi_agent_pairs()

**Status:** completed  

##### ✅ Implement get_routing_config()

**Status:** completed  


### Verification (2/2 tasks)

**Blocked by:** phase-2-files  

#### ✅ Config extension functions work

**Status:** completed  
**Type:** auto  

**Command:**
```bash
python -c 'from claude_skills.common.ai_config import get_multi_agent_pairs, get_routing_config'
```

**Expected:** Functions import successfully

#### ✅ Existing ai_config tests still pass

**Status:** completed  
**Type:** auto  

**Command:**
```bash
pytest tests/ -k ai_config -v
```

**Expected:** No regressions in existing config tests


## Migration - code-doc (7/7 tasks, 100%)

**Purpose:** Migrate code-doc skill to use shared utilities (simplest migration to validate approach)  
**Risk Level:** medium  
**Estimated Hours:** 4  

**Blocked by:** phase-2  

### File Modifications (5/5 tasks)

#### ✅ Analyze code-doc consultation code for migration

**Status:** completed  
**Estimated:** 0.5 hours  

#### ✅ src/claude_skills/claude_skills/code_doc/ai_consultation.py

**File:** `src/claude_skills/claude_skills/code_doc/ai_consultation.py`  
**Status:** completed  
**Estimated:** 2.5 hours  

**Blocked by:** task-3-1

##### ✅ Replace tool availability with detect_available_tools()

**Status:** completed  

##### ✅ Replace command building with build_tool_command()

**Status:** completed  

##### ✅ Replace parallel execution with execute_tools_parallel()

**Status:** completed  

#### ✅ Update code-doc imports and documentation

**File:** `task-3-3.md`  
**Status:** completed  
**Estimated:** 0.5 hours  

**Blocked by:** task-3-2


### Verification (2/2 tasks)

**Blocked by:** phase-3-files  

#### ✅ code-doc tests pass after migration

**Status:** completed  
**Type:** auto  

**Command:**
```bash
pytest tests/ -k code_doc -v
```

**Expected:** All code-doc tests pass

#### ✅ code-doc consultation works end-to-end

**Status:** completed  
**Type:** manual  

**Expected:** Can generate documentation with AI consultation


## Migration - sdd-plan-review (7/7 tasks, 100%)

**Purpose:** Migrate sdd-plan-review skill to use shared utilities  
**Risk Level:** medium  
**Estimated Hours:** 4  

**Blocked by:** phase-3  

### File Modifications (5/5 tasks)

#### ✅ Analyze sdd-plan-review for migration

**Status:** completed  
**Estimated:** 0.5 hours  

#### ✅ src/claude_skills/claude_skills/sdd_plan_review/reviewer.py

**File:** `src/claude_skills/claude_skills/sdd_plan_review/reviewer.py`  
**Status:** completed  
**Estimated:** 2.5 hours  

**Blocked by:** task-4-1

##### ✅ Replace check_tool_available() with shared version

**Status:** completed  

##### ✅ Replace call_tool() with execute_tool()

**Status:** completed  

##### ✅ Replace parallel execution with execute_tools_parallel()

**Status:** completed  

#### ✅ Update sdd-plan-review imports and documentation

**File:** `task-4-3.md`  
**Status:** completed  
**Estimated:** 0.5 hours  

**Blocked by:** task-4-2


### Verification (2/2 tasks)

**Blocked by:** phase-4-files  

#### ✅ sdd-plan-review tests pass after migration

**Status:** completed  
**Type:** auto  

**Command:**
```bash
pytest tests/ -k sdd_plan_review -v
```

**Expected:** All sdd-plan-review tests pass

#### ✅ sdd-plan-review works end-to-end

**Status:** completed  
**Type:** manual  

**Expected:** Can review specs with multi-model consultation


## Migration - run-tests (9/9 tasks, 100%)

**Purpose:** Migrate run-tests skill to use shared utilities (most complex due to existing config.yaml)  
**Risk Level:** high  
**Estimated Hours:** 6  

**Blocked by:** phase-4  

### File Modifications (7/7 tasks)

#### ✅ Analyze run-tests for migration complexity

**Status:** completed  
**Estimated:** 1 hours  

#### ✅ src/claude_skills/claude_skills/run_tests/consultation.py

**File:** `src/claude_skills/claude_skills/run_tests/consultation.py`  
**Status:** completed  
**Estimated:** 3.5 hours  

**Blocked by:** task-5-1

##### ✅ Replace command building with build_tool_command()

**Status:** completed  

##### ✅ Replace run_consultation() with execute_tool()

**Status:** completed  

##### ✅ Replace consult_multi_agent() with execute_tools_parallel()

**Status:** completed  

##### ✅ Migrate config loading to use ai_config.py

**Status:** completed  

#### ✅ Delete src/claude_skills/claude_skills/run_tests/tool_checking.py

**File:** `src/claude_skills/claude_skills/run_tests/tool_checking.py`  
**Status:** completed  
**Estimated:** 0.25 hours  

**Blocked by:** task-5-2

#### ✅ Update run-tests imports and documentation

**File:** `task-5-4.md`  
**Status:** completed  
**Estimated:** 0.5 hours  

**Blocked by:** task-5-3


### Verification (2/2 tasks)

**Blocked by:** phase-5-files  

#### ✅ run-tests tests pass after migration

**Status:** completed  
**Type:** auto  

**Command:**
```bash
pytest tests/ -k run_tests -v
```

**Expected:** All run-tests tests pass (CRITICAL - most heavily used skill)

#### ✅ run-tests consultation works end-to-end

**Status:** completed  
**Type:** manual  

**Expected:** Can debug test failures with AI consultation


## Testing & Verification (12/12 tasks, 100%)

**Purpose:** Comprehensive testing to ensure all skills work correctly with shared infrastructure  
**Risk Level:** low  
**Estimated Hours:** 11  

**Blocked by:** phase-5  

### File Modifications (8/8 tasks)

#### ✅ tests/integration/test_ai_tools_integration.py

**File:** `tests/integration/test_ai_tools_integration.py`  
**Status:** completed  
**Estimated:** 3 hours  

##### ✅ Test tool detection and command building

**Status:** completed  

##### ✅ Test parallel execution and timeout handling

**Status:** completed  

#### ✅ Test edge cases with malformed tool output

**File:** `task-6-2.md`  
**Status:** completed  
**Estimated:** 2 hours  

#### ✅ Test code-doc with actual AI tools

**File:** `task-6-3.md`  
**Status:** completed  
**Estimated:** 1.5 hours  

#### ✅ Test sdd-plan-review with actual AI tools

**File:** `task-6-4.md`  
**Status:** completed  
**Estimated:** 1.5 hours  

#### ✅ Test run-tests with actual AI tools

**File:** `task-6-5.md`  
**Status:** completed  
**Estimated:** 1.5 hours  

#### ✅ Update documentation for shared utilities

**File:** `task-6-6.md`  
**Status:** completed  
**Estimated:** 2 hours  

##### ✅ Create API documentation for common/ai_tools.py

**Status:** completed  

##### ✅ Update run-tests, sdd-plan-review, code-doc documentation

**Status:** completed  


### Verification (4/4 tasks)

**Blocked by:** phase-6-files  

#### ✅ All integration tests pass

**Status:** completed  
**Type:** auto  

**Command:**
```bash
pytest tests/integration/test_ai_tools_integration.py -v
```

**Expected:** All integration tests pass

#### ✅ Full test suite passes

**Status:** completed  
**Type:** auto  

**Command:**
```bash
pytest tests/ -v
```

**Expected:** All tests pass including migrated skills

#### ✅ No performance regressions

**Status:** completed  
**Type:** manual  

**Expected:** Consultation performance comparable or better than before

#### ✅ Documentation is complete and accurate

**Status:** completed  
**Type:** manual  

**Expected:** API docs and skill docs clearly explain shared utilities
