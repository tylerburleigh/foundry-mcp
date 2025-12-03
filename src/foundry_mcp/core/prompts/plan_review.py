"""
Prompt templates for SDD plan review workflow.

This module provides prompts for reviewing and critiquing
Specification-Driven Development (SDD) specifications,
identifying risks, suggesting improvements, and validating
technical approaches.

Prompt IDs:
    - review_spec: Comprehensive review of a specification
    - review_phase: Review a specific phase within a spec
    - security_review: Security-focused review of implementation plan
    - feasibility_review: Assess technical feasibility and risks
    - quick_review: Fast, high-level review for obvious issues
"""

from __future__ import annotations

import json
from typing import Any, Dict

from foundry_mcp.core.prompts import PromptBuilder


# =============================================================================
# Prompt Templates
# =============================================================================


REVIEW_SPEC_TEMPLATE = """Review the following SDD specification and provide comprehensive feedback.

## Specification Overview
- Spec ID: {spec_id}
- Title: {spec_title}
- Status: {spec_status}

## Specification Content
{spec_content}

## Review Criteria
1. **Clarity**: Are requirements and tasks clearly defined?
2. **Completeness**: Are all necessary tasks included? Any gaps?
3. **Dependencies**: Are task dependencies correctly identified?
4. **Estimates**: Are time estimates reasonable?
5. **Testability**: Can success criteria be objectively verified?
6. **Risks**: What technical or organizational risks exist?

## Output Format
Provide structured feedback in the following JSON format:
```json
{{
    "overall_assessment": "pass|needs_revision|major_concerns",
    "score": 0-100,
    "findings": [
        {{
            "category": "clarity|completeness|dependencies|estimates|testability|risks",
            "severity": "info|warning|error",
            "location": "task-id or 'general'",
            "issue": "description of the issue",
            "suggestion": "how to address it"
        }}
    ],
    "summary": "brief overall summary"
}}
```
"""


REVIEW_PHASE_TEMPLATE = """Review the following phase from an SDD specification.

## Phase Information
- Spec ID: {spec_id}
- Phase ID: {phase_id}
- Phase Title: {phase_title}
- Phase Status: {phase_status}

## Phase Content
{phase_content}

## Context
- Previous Phases Completed: {completed_phases}
- Dependencies from Other Phases: {cross_phase_deps}

## Review Criteria
1. **Scope**: Is the phase scope well-defined and achievable?
2. **Task Breakdown**: Are tasks appropriately sized and ordered?
3. **Dependencies**: Are internal dependencies clear?
4. **Verification**: Does the phase include adequate verification tasks?
5. **Blockers**: Any potential blockers or risks?

## Output Format
Provide feedback in the following JSON format:
```json
{{
    "phase_assessment": "ready|needs_work|blocked",
    "findings": [
        {{
            "task_id": "task-id or null",
            "severity": "info|warning|error",
            "issue": "description",
            "suggestion": "recommended action"
        }}
    ],
    "recommended_actions": ["list of actions before proceeding"],
    "summary": "brief phase summary"
}}
```
"""


SECURITY_REVIEW_TEMPLATE = """Perform a security-focused review of the following implementation plan.

## Specification
- Spec ID: {spec_id}
- Title: {spec_title}

## Implementation Plan
{spec_content}

## Security Review Focus Areas
1. **Input Validation**: Are all inputs properly validated and sanitized?
2. **Authentication/Authorization**: Are access controls properly planned?
3. **Data Protection**: Is sensitive data properly handled and protected?
4. **Dependencies**: Are third-party dependencies secure and up-to-date?
5. **Error Handling**: Does error handling avoid information leakage?
6. **Injection Risks**: Are there SQL, command, or other injection risks?
7. **OWASP Top 10**: Consider OWASP Top 10 vulnerabilities

## Output Format
Provide security findings in the following JSON format:
```json
{{
    "security_score": 0-100,
    "risk_level": "low|medium|high|critical",
    "findings": [
        {{
            "category": "input_validation|auth|data_protection|dependencies|error_handling|injection|other",
            "severity": "info|low|medium|high|critical",
            "location": "task-id or file reference",
            "vulnerability": "description of the security concern",
            "remediation": "recommended fix or mitigation",
            "cwe_id": "CWE-XXX if applicable"
        }}
    ],
    "recommendations": ["prioritized list of security improvements"],
    "summary": "overall security assessment"
}}
```
"""


FEASIBILITY_REVIEW_TEMPLATE = """Assess the technical feasibility of the following implementation plan.

## Specification
- Spec ID: {spec_id}
- Title: {spec_title}
- Total Estimated Hours: {total_hours}

## Implementation Plan
{spec_content}

## Constraints
- Team Size: {team_size}
- Timeline: {timeline}
- Technology Stack: {tech_stack}
- Known Constraints: {constraints}

## Feasibility Assessment Areas
1. **Technical Complexity**: Is the approach technically sound?
2. **Resource Requirements**: Are estimates realistic given constraints?
3. **Dependencies**: Are external dependencies available and stable?
4. **Skills**: Does the team have required expertise?
5. **Timeline**: Is the timeline achievable?
6. **Risk Factors**: What could cause delays or failure?

## Output Format
Provide feasibility assessment in the following JSON format:
```json
{{
    "feasibility_score": 0-100,
    "confidence": "low|medium|high",
    "assessment": "feasible|feasible_with_risks|not_feasible",
    "findings": [
        {{
            "area": "complexity|resources|dependencies|skills|timeline|risk",
            "severity": "info|concern|blocker",
            "issue": "description of the concern",
            "mitigation": "suggested approach to address"
        }}
    ],
    "critical_path": ["list of tasks on critical path"],
    "risk_factors": ["prioritized list of risk factors"],
    "recommendations": ["suggestions for improving feasibility"],
    "summary": "overall feasibility assessment"
}}
```
"""


QUICK_REVIEW_TEMPLATE = """Perform a quick review of the following SDD specification for obvious issues.

## Specification
- Spec ID: {spec_id}
- Title: {spec_title}
- Task Count: {task_count}

## Specification Summary
{spec_summary}

## Quick Review Checklist
- [ ] Spec has clear objective
- [ ] Tasks have descriptions
- [ ] Dependencies are defined
- [ ] Verification tasks exist
- [ ] No circular dependencies
- [ ] Reasonable estimates

## Output Format
Provide a quick assessment in JSON format:
```json
{{
    "status": "ok|issues_found",
    "issues": [
        {{
            "type": "missing_objective|missing_description|missing_deps|missing_verification|circular_deps|unrealistic_estimates|other",
            "severity": "warning|error",
            "message": "brief description"
        }}
    ],
    "ready_to_proceed": true|false,
    "summary": "one-line summary"
}}
```
"""


# =============================================================================
# Template Registry
# =============================================================================


TEMPLATES = {
    "review_spec": REVIEW_SPEC_TEMPLATE,
    "review_phase": REVIEW_PHASE_TEMPLATE,
    "security_review": SECURITY_REVIEW_TEMPLATE,
    "feasibility_review": FEASIBILITY_REVIEW_TEMPLATE,
    "quick_review": QUICK_REVIEW_TEMPLATE,
}


# =============================================================================
# Prompt Builder Implementation
# =============================================================================


class PlanReviewPromptBuilder(PromptBuilder):
    """
    Prompt builder for SDD plan review workflow.

    Provides templates for reviewing specifications, phases,
    and conducting specialized reviews (security, feasibility).
    """

    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a plan review prompt.

        Args:
            prompt_id: One of: review_spec, review_phase, security_review,
                      feasibility_review, quick_review
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

        # Handle spec_content specially - may be dict or string
        spec_content = context.get("spec_content", "")
        if isinstance(spec_content, dict):
            spec_content = json.dumps(spec_content, indent=2)

        phase_content = context.get("phase_content", "")
        if isinstance(phase_content, dict):
            phase_content = json.dumps(phase_content, indent=2)

        # Provide safe defaults for optional context keys
        safe_context = {
            "spec_id": context.get("spec_id", "unknown"),
            "spec_title": context.get("spec_title", "Unknown Specification"),
            "spec_status": context.get("spec_status", "unknown"),
            "spec_content": spec_content,
            "spec_summary": context.get("spec_summary", "No summary provided"),
            "phase_id": context.get("phase_id", "unknown"),
            "phase_title": context.get("phase_title", "Unknown Phase"),
            "phase_status": context.get("phase_status", "unknown"),
            "phase_content": phase_content,
            "completed_phases": context.get("completed_phases", "None"),
            "cross_phase_deps": context.get("cross_phase_deps", "None identified"),
            "total_hours": context.get("total_hours", "Not estimated"),
            "team_size": context.get("team_size", "Not specified"),
            "timeline": context.get("timeline", "Not specified"),
            "tech_stack": context.get("tech_stack", "Not specified"),
            "constraints": context.get("constraints", "None specified"),
            "task_count": context.get("task_count", 0),
        }

        try:
            return template.format(**safe_context)
        except KeyError as exc:
            raise ValueError(f"Missing required context key: {exc}") from exc

    def list_prompts(self) -> list[str]:
        """Return available prompt IDs for plan review."""
        return sorted(TEMPLATES.keys())


__all__ = [
    "PlanReviewPromptBuilder",
    "TEMPLATES",
]
