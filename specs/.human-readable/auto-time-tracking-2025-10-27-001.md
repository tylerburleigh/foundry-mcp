# Automatic Time Tracking Based on Timestamps

**Spec ID:** `auto-time-tracking-2025-10-27-001`  
**Status:** pending (0/35 tasks, 0%)  
**Estimated Effort:** 24 hours  
**Complexity:** medium  

Implement automatic time tracking that calculates actual_hours from started_at and completed_at timestamps instead of requiring manual entry

## Core Time Calculation Logic (0/5 tasks, 0%)

**Purpose:** Add timestamp parsing and duration calculation utilities  
**Risk Level:** low  
**Estimated Hours:** 3  


### File Modifications (0/3 tasks)

#### ⏳ src/claude_skills/claude_skills/sdd_update/time_tracking.py

**File:** `src/claude_skills/claude_skills/sdd_update/time_tracking.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add calculate_time_from_timestamps() function with ISO 8601 parsing and duration calculation  

##### ⏳ Add calculate_time_from_timestamps() function

**Status:** pending  

##### ⏳ Add error handling for invalid timestamps

**Status:** pending  

**Depends on:** task-1-1-1

##### ⏳ Add rounding logic for decimal hours

**Status:** pending  

**Depends on:** task-1-1-1


### Verification (0/2 tasks)

**Blocked by:** phase-1-files  

#### ⏳ Test time calculation with known timestamps

**Status:** pending  
**Type:** manual  

**Expected:** Function correctly calculates 2.5 hours for 2025-01-01T10:00:00Z to 2025-01-01T12:30:00Z

#### ⏳ Test error handling for edge cases

**Status:** pending  
**Type:** manual  

**Expected:** Function handles None, invalid format, and reversed timestamps gracefully


## Task-Level Auto-Tracking (0/11 tasks, 0%)

**Purpose:** Integrate automatic time calculation into task completion workflow and clean up manual time entry  
**Risk Level:** medium  
**Estimated Hours:** 8  

**Blocked by:** phase-1  

### File Modifications (0/8 tasks)

#### ⏳ src/claude_skills/claude_skills/sdd_update/status.py

**File:** `src/claude_skills/claude_skills/sdd_update/status.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Modify update_task_status() to reset started_at on EVERY transition to in_progress, not just first time  

##### ⏳ Update started_at logic in update_task_status()

**Status:** pending  

##### ⏳ Ensure timestamp format consistency

**Status:** pending  

**Depends on:** task-2-1-1

#### ⏳ src/claude_skills/claude_skills/sdd_update/workflow.py

**File:** `src/claude_skills/claude_skills/sdd_update/workflow.py`  
**Status:** pending  
**Estimated:** 3 hours  
**Changes:** Update complete_task_workflow() to auto-calculate actual_hours from timestamps  

**Depends on:** task-2-1

##### ⏳ Add automatic time calculation in complete_task_workflow()

**Status:** pending  

##### ⏳ Handle missing timestamp edge cases

**Status:** pending  

**Depends on:** task-2-2-1

##### ⏳ Update track_time() call to use calculated value

**Status:** pending  

**Depends on:** task-2-2-1

#### ⏳ src/claude_skills/claude_skills/sdd_update/cli.py

**File:** `src/claude_skills/claude_skills/sdd_update/cli.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Remove --actual-hours flag and track-time command (clean break)  

##### ⏳ Remove --actual-hours flag from complete-task command

**Status:** pending  

##### ⏳ Remove track-time command

**Status:** pending  

##### ⏳ Update CLI help text for complete-task

**Status:** pending  

**Depends on:** task-2-3-1


### Verification (0/3 tasks)

**Blocked by:** phase-2-files  

#### ⏳ Test task status transition resets started_at

**Status:** pending  
**Type:** manual  

**Command:**
```bash
sdd update-status test-spec task-1-1 in_progress (multiple times)
```

**Expected:** started_at is updated each time status moves to in_progress

#### ⏳ Test automatic time calculation on task completion

**Status:** pending  
**Type:** manual  

**Command:**
```bash
sdd complete-task test-spec task-1-1
```

**Expected:** actual_hours is automatically populated in task metadata based on timestamp difference

#### ⏳ Verify CLI commands no longer accept --actual-hours

**Status:** pending  
**Type:** manual  

**Command:**
```bash
sdd complete-task --help
```

**Expected:** No --actual-hours flag listed; track-time command not found


## Spec-Level Time Aggregation (0/6 tasks, 0%)

**Purpose:** Automatically calculate total spec time from all task times when spec completes  
**Risk Level:** low  
**Estimated Hours:** 4  

**Blocked by:** phase-2  

### File Modifications (0/4 tasks)

#### ⏳ src/claude_skills/claude_skills/sdd_update/time_tracking.py

**File:** `src/claude_skills/claude_skills/sdd_update/time_tracking.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add aggregate_task_times() function to sum all task actual_hours  

##### ⏳ Add aggregate_task_times() function

**Status:** pending  

##### ⏳ Handle partial completion scenarios

**Status:** pending  

**Depends on:** task-3-1-1

#### ⏳ src/claude_skills/claude_skills/sdd_update/lifecycle.py

**File:** `src/claude_skills/claude_skills/sdd_update/lifecycle.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Update complete_spec() to auto-calculate and store spec-level actual_hours  

**Depends on:** task-3-1

##### ⏳ Call aggregate_task_times() in complete_spec()

**Status:** pending  

##### ⏳ Remove --actual-hours parameter from complete-spec CLI

**Status:** pending  

**Depends on:** task-3-2-1


### Verification (0/2 tasks)

**Blocked by:** phase-3-files  

#### ⏳ Test spec-level time aggregation

**Status:** pending  
**Type:** manual  

**Expected:** Complete spec with 3 tasks (2h, 3h, 1.5h) should automatically calculate spec actual_hours as 6.5

#### ⏳ Test partial completion handling

**Status:** pending  
**Type:** manual  

**Expected:** Spec with some tasks missing actual_hours should sum only tasks with times (no errors)


## Schema & Documentation Updates (0/5 tasks, 0%)

**Purpose:** Update schema and document new automatic tracking behavior  
**Risk Level:** low  
**Estimated Hours:** 4  

**Blocked by:** phase-3  

### File Modifications (0/4 tasks)

#### ⏳ src/claude_skills/schemas/sdd-spec-schema.json

**File:** `src/claude_skills/schemas/sdd-spec-schema.json`  
**Status:** pending  
**Estimated:** 1 hours  
**Changes:** Formally define timestamp and actual_hours fields in schema  

##### ⏳ Add timestamp fields to metadata schema

**Status:** pending  

#### ⏳ skills/sdd-update/SKILL.md

**File:** `skills/sdd-update/SKILL.md`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Update time tracking documentation to explain automatic behavior  

##### ⏳ Update time tracking section

**Status:** pending  

##### ⏳ Add workflow examples with automatic time tracking

**Status:** pending  

**Depends on:** task-4-2-1

#### ⏳ docs/DOCUMENTATION.md

**File:** `docs/DOCUMENTATION.md`  
**Status:** pending  
**Estimated:** 1 hours  
**Changes:** Update main documentation with automatic time tracking examples  

##### ⏳ Update complete-task examples

**Status:** pending  


### Verification (0/1 tasks)

**Blocked by:** phase-4-files  

#### ⏳ Documentation accuracy review

**Status:** pending  
**Type:** manual  

**Expected:** All documentation accurately reflects new automatic tracking behavior with no references to removed commands


## Testing & Verification (0/8 tasks, 0%)

**Purpose:** Comprehensive testing of automatic time tracking  
**Risk Level:** medium  
**Estimated Hours:** 5  

**Blocked by:** phase-2  

### File Modifications (0/6 tasks)

#### ⏳ src/claude_skills/claude_skills/tests/unit/test_sdd_update/test_time_tracking.py

**File:** `src/claude_skills/claude_skills/tests/unit/test_sdd_update/test_time_tracking.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add unit tests for time calculation and aggregation functions  

##### ⏳ Test calculate_time_from_timestamps() with valid inputs

**Status:** pending  

##### ⏳ Test edge cases and error handling

**Status:** pending  

##### ⏳ Test aggregate_task_times() function

**Status:** pending  

#### ⏳ src/claude_skills/claude_skills/tests/unit/test_sdd_update/test_workflow.py

**File:** `src/claude_skills/claude_skills/tests/unit/test_sdd_update/test_workflow.py`  
**Status:** pending  
**Estimated:** 2 hours  
**Changes:** Add integration tests for complete_task_workflow with automatic time tracking  

##### ⏳ Test complete_task_workflow auto-calculates time

**Status:** pending  

##### ⏳ Test workflow handles missing timestamps gracefully

**Status:** pending  

#### ⏳ src/claude_skills/claude_skills/tests/unit/test_sdd_update/test_status.py

**File:** `src/claude_skills/claude_skills/tests/unit/test_sdd_update/test_status.py`  
**Status:** pending  
**Estimated:** 1 hours  
**Changes:** Add tests for started_at reset behavior  

##### ⏳ Test started_at resets on each in_progress transition

**Status:** pending  


### Verification (0/2 tasks)

**Blocked by:** phase-5-files  

#### ⏳ Run full test suite

**Status:** pending  
**Type:** auto  

**Command:**
```bash
python -m pytest src/claude_skills/claude_skills/tests/unit/test_sdd_update/
```

**Expected:** All tests pass with no failures

#### ⏳ Test backward compatibility with existing specs

**Status:** pending  
**Type:** manual  

**Expected:** Existing specs in specs/ directory can still be loaded and processed without errors
