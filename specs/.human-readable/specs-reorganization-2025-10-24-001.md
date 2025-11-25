# Specs Directory Reorganization

**Spec ID:** `specs-reorganization-2025-10-24-001`  
**Status:** pending (0/22 tasks, 0%)  
**Estimated Effort:** 4 hours  
**Complexity:** low  

Separate generated metadata files (validation reports, review outputs, backups) from actual spec JSON files by creating dedicated hidden directories and updating all CLI code to use new paths.

## Directory Structure & Gitignore (0/5 tasks, 0%)

**Purpose:** Create new directory structure and update version control settings  
**Risk Level:** low  
**Estimated Hours:** 0.5  


### File Modifications (0/3 tasks)

#### ⏳ Create specs/.reports/ directory

**File:** `specs/.reports/`  
**Status:** pending  
**Changes:** mkdir -p specs/.reports && add README.md explaining purpose  

#### ⏳ Create specs/.reviews/ directory

**File:** `specs/.reviews/`  
**Status:** pending  
**Changes:** mkdir -p specs/.reviews && add README.md explaining purpose  

#### ⏳ Create specs/.backups/ directory

**File:** `specs/.backups/`  
**Status:** pending  
**Changes:** mkdir -p specs/.backups && add README.md explaining purpose  


### Verification (0/2 tasks)

**Blocked by:** phase-1-files  

#### ⏳ Directories exist and are accessible

**Status:** pending  
**Type:** auto  

**Command:**
```bash
ls -la specs/.reports specs/.reviews specs/.backups
```

**Expected:** All three directories exist with README files

#### ⏳ Gitignore prevents tracking

**Status:** pending  
**Type:** auto  

**Command:**
```bash
git status specs/.reports specs/.reviews specs/.backups
```

**Expected:** Directories not shown in git status (ignored)


## Update Validation Report Paths (0/3 tasks, 0%)

**Purpose:** Modify sdd-validate to save reports to specs/.reports/  
**Risk Level:** low  
**Estimated Hours:** 1  

**Blocked by:** phase-1  

### File Modifications (0/2 tasks)

#### ⏳ src/claude_skills/claude_skills/sdd_validate/cli.py

**File:** `src/claude_skills/claude_skills/sdd_validate/cli.py`  
**Status:** pending  
**Estimated:** 0.5 hours  

**Blocked by:** task-1-1

##### ⏳ Update cmd_validate report path (line 219)

**Status:** pending  
**Changes:** report_file = Path('specs/.reports') / f"{spec_file.stem}-validation-report{suffix}"  

##### ⏳ Update cmd_report default output path (lines 400-404)

**Status:** pending  
**Changes:** output_path = Path(args.output) if args.output else Path('specs/.reports') / spec_file.stem  


### Verification (0/1 tasks)

**Blocked by:** phase-2-files  

#### ⏳ Validation reports save to .reports/

**Status:** pending  
**Type:** auto  

**Command:**
```bash
sdd validate specs/active/test-spec.json --report && ls specs/.reports/
```

**Expected:** Report file appears in specs/.reports/ directory


## Update Backup File Paths (0/2 tasks, 0%)

**Purpose:** Modify sdd-validate to save backups to specs/.backups/  
**Risk Level:** low  
**Estimated Hours:** 0.5  

**Blocked by:** phase-1  

### File Modifications (0/1 tasks)

#### ⏳ src/claude_skills/claude_skills/sdd_validate/fix.py

**File:** `src/claude_skills/claude_skills/sdd_validate/fix.py`  
**Status:** pending  
**Estimated:** 0.5 hours  

**Blocked by:** task-1-3

##### ⏳ Update backup path generation (line 131)

**Status:** pending  
**Changes:** backup_path = Path('specs/.backups') / f"{Path(spec_path).stem}.json.backup"  


### Verification (0/1 tasks)

**Blocked by:** phase-3-files  

#### ⏳ Backup files save to .backups/

**Status:** pending  
**Type:** auto  

**Command:**
```bash
sdd fix specs/active/test-spec.json && ls specs/.backups/
```

**Expected:** Backup file appears in specs/.backups/ directory


## Update Review Output Paths (0/2 tasks, 0%)

**Purpose:** Modify sdd-plan-review to save reviews to specs/.reviews/  
**Risk Level:** low  
**Estimated Hours:** 0.5  

**Blocked by:** phase-1  

### File Modifications (0/1 tasks)

#### ⏳ src/claude_skills/claude_skills/sdd_plan_review/cli.py

**File:** `src/claude_skills/claude_skills/sdd_plan_review/cli.py`  
**Status:** pending  
**Estimated:** 0.5 hours  

**Blocked by:** task-1-2

##### ⏳ Update default output path (lines 132-154)

**Status:** pending  
**Changes:** Add logic to default output_path to specs/.reviews/{spec_id}-review{.json|.md}  


### Verification (0/1 tasks)

**Blocked by:** phase-4-files  

#### ⏳ Review outputs save to .reviews/

**Status:** pending  
**Type:** manual  

**Command:**
```bash
Run review on test spec and verify output location
```

**Expected:** Review files appear in specs/.reviews/ directory


## Migration & Testing (0/10 tasks, 0%)

**Purpose:** Move existing metadata files and verify all functionality works  
**Risk Level:** medium  
**Estimated Hours:** 1.5  

**Blocked by:** phase-2, phase-3, phase-4  

### File Modifications (0/3 tasks)

#### ⏳ Move validation reports to .reports/

**File:** `specs/active/*-validation-report*.md`  
**Status:** pending  
**Changes:** mv specs/active/*-validation-report*.md specs/.reports/  

**Blocked by:** task-1-1, task-2-1

#### ⏳ Move review files to .reviews/

**File:** `specs/active/*-review.*`  
**Status:** pending  
**Changes:** mv specs/active/*-review.* specs/.reviews/  

**Blocked by:** task-1-2, task-4-1

#### ⏳ Move backup files to .backups/

**File:** `specs/active/*.json.backup`  
**Status:** pending  
**Changes:** mv specs/active/*.json.backup specs/.backups/  

**Blocked by:** task-1-3, task-3-1


### Verification (0/7 tasks)

**Blocked by:** phase-5-files  

#### ⏳ specs/active/ is clean

**Status:** pending  
**Type:** auto  

**Command:**
```bash
ls specs/active/ | grep -E '(validation-report|review|backup)'
```

**Expected:** No metadata files found in specs/active/

#### ⏳ Validation report generation works

**Status:** pending  
**Type:** auto  

**Command:**
```bash
sdd validate specs/active/specs-reorganization-2025-10-24-001.json --report
```

**Expected:** Report created in specs/.reports/

#### ⏳ Backup file creation works

**Status:** pending  
**Type:** auto  

**Command:**
```bash
sdd fix specs/active/specs-reorganization-2025-10-24-001.json --preview && check backup location
```

**Expected:** Backup created in specs/.backups/ if fixes applied

#### ⏳ Custom --output flag still works

**Status:** pending  
**Type:** manual  

**Command:**
```bash
sdd report specs/active/test.json --output /tmp/test-report.md
```

**Expected:** Report created at specified custom path, not default .reports/

#### ⏳ All migrated files accessible

**Status:** pending  
**Type:** auto  

**Command:**
```bash
ls -R specs/.reports/ specs/.reviews/ specs/.backups/
```

**Expected:** All previously moved files present in new locations

#### ⏳ Gitignore working correctly

**Status:** pending  
**Type:** auto  

**Command:**
```bash
git status | grep -E '(.reports|.reviews|.backups)'
```

**Expected:** No metadata directories appear in git status

#### ⏳ End-to-end workflow test

**Status:** pending  
**Type:** manual  

**Command:**
```bash
Create spec -> validate with report -> fix with backup -> verify locations
```

**Expected:** All outputs in correct hidden directories, specs/active/ remains clean
