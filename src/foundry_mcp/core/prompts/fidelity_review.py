"""
Prompt templates for fidelity review workflow.

This module provides prompts for comparing implementation against
specifications, identifying deviations, and assessing compliance
with documented requirements.

Prompt IDs:
    - review_task: Compare task implementation against spec requirements
    - review_phase: Review entire phase for fidelity to spec
    - compare_files: Compare specific files against spec expectations
    - deviation_analysis: Analyze identified deviations for impact
    - compliance_summary: Generate compliance summary report
"""

from __future__ import annotations

import json
from typing import Any, Dict

from foundry_mcp.core.prompts import PromptBuilder


# =============================================================================
# Prompt Templates
# =============================================================================


REVIEW_TASK_TEMPLATE = """Compare the task implementation against its specification requirements.

## Task Specification
- Spec ID: {spec_id}
- Task ID: {task_id}
- Task Title: {task_title}
- Expected File(s): {expected_files}

## Task Requirements
{task_requirements}

## Actual Implementation
### File: {file_path}
```{file_type}
{file_content}
```

## Test Results (if available)
{test_results}

## Fidelity Review Requirements
1. **Requirement Coverage**: Are all specified requirements implemented?
2. **Implementation Quality**: Does the implementation meet quality standards?
3. **Deviations**: Any deviations from the specification?
4. **Missing Elements**: Any required elements not implemented?
5. **Additional Changes**: Any changes beyond specification scope?

## Output Format
Provide fidelity assessment in the following JSON format:
```json
{{
    "task_id": "{task_id}",
    "fidelity_score": 0-100,
    "verdict": "compliant|partial|non_compliant",
    "requirements_status": [
        {{
            "requirement": "description",
            "status": "implemented|partial|missing|deviated",
            "evidence": "location or description of implementation",
            "notes": "any relevant notes"
        }}
    ],
    "deviations": [
        {{
            "type": "missing|added|changed|moved",
            "description": "what deviated from spec",
            "location": "file:line or general location",
            "severity": "minor|moderate|major",
            "impact": "impact assessment",
            "recommendation": "suggested action"
        }}
    ],
    "summary": "brief overall assessment"
}}
```
"""


REVIEW_PHASE_TEMPLATE = """Review the entire phase implementation for fidelity to specification.

## Phase Specification
- Spec ID: {spec_id}
- Phase ID: {phase_id}
- Phase Title: {phase_title}
- Expected Tasks: {expected_task_count}
- Completed Tasks: {completed_task_count}

## Phase Requirements
{phase_requirements}

## Implementation Summary
{implementation_summary}

## File Changes in Phase
{file_changes}

## Phase Fidelity Assessment
1. **Task Completion**: Are all phase tasks completed?
2. **Requirement Coverage**: Are phase requirements satisfied?
3. **Integration**: Do components integrate correctly?
4. **Quality**: Does implementation meet quality standards?
5. **Documentation**: Is implementation properly documented?

## Output Format
Provide phase fidelity assessment in the following JSON format:
```json
{{
    "phase_id": "{phase_id}",
    "fidelity_score": 0-100,
    "verdict": "compliant|partial|non_compliant",
    "task_status": [
        {{
            "task_id": "task-id",
            "title": "task title",
            "status": "compliant|partial|non_compliant|skipped",
            "notes": "brief notes"
        }}
    ],
    "phase_deviations": [
        {{
            "type": "scope_creep|missing_functionality|integration_issue|quality_issue",
            "description": "description of deviation",
            "affected_tasks": ["task-ids"],
            "severity": "minor|moderate|major",
            "recommendation": "suggested action"
        }}
    ],
    "integration_assessment": "notes on how well components work together",
    "summary": "overall phase assessment"
}}
```
"""


COMPARE_FILES_TEMPLATE = """Compare the actual file implementation against specification expectations.

## File Information
- File Path: {file_path}
- Spec ID: {spec_id}
- Related Task(s): {related_tasks}

## Expected Content/Structure
{expected_content}

## Actual Content
```{file_type}
{actual_content}
```

## Comparison Requirements
1. **Structure Match**: Does file structure match expectations?
2. **API Compliance**: Do public interfaces match specification?
3. **Behavior**: Does implementation behavior match requirements?
4. **Naming**: Do identifiers follow specified naming conventions?
5. **Completeness**: Are all expected elements present?

## Output Format
Provide file comparison in the following JSON format:
```json
{{
    "file_path": "{file_path}",
    "match_percentage": 0-100,
    "structural_match": true|false,
    "api_compliant": true|false,
    "differences": [
        {{
            "category": "structure|api|behavior|naming|missing|extra",
            "expected": "what was expected",
            "actual": "what was found",
            "location": "line or section",
            "severity": "info|warning|error",
            "fix_suggestion": "how to align with spec"
        }}
    ],
    "summary": "brief comparison summary"
}}
```
"""


DEVIATION_ANALYSIS_TEMPLATE = """Analyze the impact and implications of identified specification deviations.

## Context
- Spec ID: {spec_id}
- Scope: {scope}

## Identified Deviations
{deviations_json}

## Original Requirements
{original_requirements}

## Analysis Requirements
1. **Impact Assessment**: What is the impact of each deviation?
2. **Risk Analysis**: What risks do these deviations introduce?
3. **Dependency Effects**: How do deviations affect dependent tasks?
4. **Remediation Options**: How can deviations be addressed?
5. **Acceptance Criteria**: Which deviations might be acceptable?

## Output Format
Provide deviation analysis in the following JSON format:
```json
{{
    "analysis_scope": "{scope}",
    "total_deviations": 0,
    "critical_count": 0,
    "deviation_analysis": [
        {{
            "deviation_id": "index or identifier",
            "original_deviation": "brief description",
            "impact_assessment": {{
                "functional_impact": "none|minor|moderate|major",
                "security_impact": "none|minor|moderate|major",
                "performance_impact": "none|minor|moderate|major",
                "maintenance_impact": "none|minor|moderate|major"
            }},
            "affected_components": ["list of affected components"],
            "downstream_effects": ["list of downstream effects"],
            "remediation_options": [
                {{
                    "option": "description",
                    "effort": "low|medium|high",
                    "risk": "low|medium|high"
                }}
            ],
            "recommendation": "accept|fix_now|fix_later|needs_discussion",
            "rationale": "explanation for recommendation"
        }}
    ],
    "overall_risk_level": "low|medium|high|critical",
    "recommended_actions": ["prioritized list of recommended actions"],
    "summary": "overall deviation analysis summary"
}}
```
"""


COMPLIANCE_SUMMARY_TEMPLATE = """Generate a compliance summary report for the specification.

## Specification
- Spec ID: {spec_id}
- Title: {spec_title}
- Total Phases: {total_phases}
- Total Tasks: {total_tasks}

## Fidelity Review Data
{review_data}

## Summary Requirements
1. **Overall Compliance**: What is the overall compliance level?
2. **Phase Breakdown**: Compliance by phase
3. **Critical Issues**: List critical compliance issues
4. **Recommendations**: Prioritized recommendations
5. **Sign-off Status**: Is the implementation ready for approval?

## Output Format
Provide compliance summary in the following JSON format:
```json
{{
    "spec_id": "{spec_id}",
    "spec_title": "{spec_title}",
    "overall_compliance": {{
        "score": 0-100,
        "status": "compliant|mostly_compliant|needs_work|non_compliant",
        "tasks_compliant": 0,
        "tasks_partial": 0,
        "tasks_non_compliant": 0
    }},
    "phase_breakdown": [
        {{
            "phase_id": "phase-id",
            "phase_title": "title",
            "compliance_score": 0-100,
            "status": "compliant|partial|non_compliant"
        }}
    ],
    "critical_issues": [
        {{
            "issue": "description",
            "location": "task or phase id",
            "priority": "p0|p1|p2",
            "remediation": "suggested fix"
        }}
    ],
    "recommendations": [
        {{
            "recommendation": "description",
            "priority": "critical|high|medium|low",
            "effort": "low|medium|high"
        }}
    ],
    "sign_off": {{
        "ready": true|false,
        "blocking_issues": ["list of issues that must be resolved"],
        "conditions": ["conditions for approval if any"]
    }},
    "summary": "executive summary of compliance status"
}}
```
"""


# =============================================================================
# Template Registry
# =============================================================================


TEMPLATES = {
    "review_task": REVIEW_TASK_TEMPLATE,
    "review_phase": REVIEW_PHASE_TEMPLATE,
    "compare_files": COMPARE_FILES_TEMPLATE,
    "deviation_analysis": DEVIATION_ANALYSIS_TEMPLATE,
    "compliance_summary": COMPLIANCE_SUMMARY_TEMPLATE,
}


# =============================================================================
# Prompt Builder Implementation
# =============================================================================


class FidelityReviewPromptBuilder(PromptBuilder):
    """
    Prompt builder for fidelity review workflow.

    Provides templates for comparing implementations against specifications
    and generating compliance reports.
    """

    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a fidelity review prompt.

        Args:
            prompt_id: One of: review_task, review_phase, compare_files,
                      deviation_analysis, compliance_summary
            context: Template context variables

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt_id is not recognized
        """
        template = TEMPLATES.get(prompt_id)
        if template is None:
            available = ", ".join(sorted(TEMPLATES.keys()))
            raise ValueError(
                f"Unknown prompt_id '{prompt_id}'. Available: {available}"
            )

        # Handle JSON content specially
        deviations_json = context.get("deviations_json", [])
        if isinstance(deviations_json, (list, dict)):
            deviations_json = json.dumps(deviations_json, indent=2)

        review_data = context.get("review_data", {})
        if isinstance(review_data, dict):
            review_data = json.dumps(review_data, indent=2)

        # Provide safe defaults for optional context keys
        safe_context = {
            "spec_id": context.get("spec_id", "unknown"),
            "spec_title": context.get("spec_title", "Unknown Specification"),
            "task_id": context.get("task_id", "unknown"),
            "task_title": context.get("task_title", "Unknown Task"),
            "task_requirements": context.get("task_requirements", "No requirements specified"),
            "expected_files": context.get("expected_files", "Not specified"),
            "file_path": context.get("file_path", "unknown"),
            "file_type": context.get("file_type", ""),
            "file_content": context.get("file_content", ""),
            "actual_content": context.get("actual_content", ""),
            "expected_content": context.get("expected_content", "Not specified"),
            "test_results": context.get("test_results", "No test results available"),
            "phase_id": context.get("phase_id", "unknown"),
            "phase_title": context.get("phase_title", "Unknown Phase"),
            "phase_requirements": context.get("phase_requirements", "No requirements specified"),
            "expected_task_count": context.get("expected_task_count", 0),
            "completed_task_count": context.get("completed_task_count", 0),
            "implementation_summary": context.get("implementation_summary", "No summary provided"),
            "file_changes": context.get("file_changes", "No changes documented"),
            "related_tasks": context.get("related_tasks", "None specified"),
            "scope": context.get("scope", "general"),
            "deviations_json": deviations_json,
            "original_requirements": context.get("original_requirements", "Not provided"),
            "total_phases": context.get("total_phases", 0),
            "total_tasks": context.get("total_tasks", 0),
            "review_data": review_data,
        }

        try:
            return template.format(**safe_context)
        except KeyError as exc:
            raise ValueError(f"Missing required context key: {exc}") from exc

    def list_prompts(self) -> list[str]:
        """Return available prompt IDs for fidelity review."""
        return sorted(TEMPLATES.keys())


__all__ = [
    "FidelityReviewPromptBuilder",
    "TEMPLATES",
]
