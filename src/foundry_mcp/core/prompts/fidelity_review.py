"""
Prompt templates for fidelity review workflow.

This module provides prompts for comparing implementation against
specifications, identifying deviations, and assessing compliance
with documented requirements.

Prompt IDs (PromptTemplate-based):
    - FIDELITY_REVIEW_V1: Main 6-section fidelity review prompt
    - FIDELITY_DEVIATION_ANALYSIS_V1: Analyze identified deviations
    - FIDELITY_COMPLIANCE_SUMMARY_V1: Generate compliance summary

Legacy Prompt IDs (string templates for backward compatibility):
    - review_task: Compare task implementation against spec requirements
    - review_phase: Review entire phase for fidelity to spec
    - compare_files: Compare specific files against spec expectations
    - deviation_analysis: Analyze identified deviations for impact
    - compliance_summary: Generate compliance summary report
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from foundry_mcp.core.prompts import PromptBuilder, PromptTemplate


# =============================================================================
# Response Schema
# =============================================================================


# JSON response schema for fidelity reviews - structured format for AI response
FIDELITY_RESPONSE_SCHEMA = """{
  "verdict": "pass|fail|partial|unknown",
  "summary": "Overall findings (any length).",
  "requirement_alignment": {
    "answer": "yes|no|partial",
    "details": "Explain how implementation aligns or diverges."
  },
  "success_criteria": {
    "met": "yes|no|partial",
    "details": "Call out verification steps passed or missing."
  },
  "deviations": [
    {
      "description": "Describe deviation from the spec.",
      "justification": "Optional rationale or evidence.",
      "severity": "critical|high|medium|low"
    }
  ],
  "test_coverage": {
    "status": "sufficient|insufficient|not_applicable",
    "details": "Summarise test evidence or gaps."
  },
  "code_quality": {
    "issues": ["Describe each notable quality concern."],
    "details": "Optional supporting commentary."
  },
  "documentation": {
    "status": "adequate|inadequate|not_applicable",
    "details": "Note doc updates or omissions."
  },
  "issues": ["Concise list of primary issues for consensus logic."],
  "recommendations": ["Actionable next steps to resolve findings."]
}"""


# =============================================================================
# Severity Categorization Keywords
# =============================================================================


# CRITICAL: Security vulnerabilities, data loss, crashes
CRITICAL_KEYWORDS = [
    'security', 'vulnerability', 'injection', 'xss', 'csrf',
    'authentication bypass', 'unauthorized access', 'data loss',
    'crash', 'segfault', 'memory leak', 'remote code execution',
    'privilege escalation', 'buffer overflow'
]

# HIGH: Incorrect behavior, spec violations, broken functionality
HIGH_KEYWORDS = [
    'incorrect', 'wrong', 'broken', 'fails', 'failure',
    'spec violation', 'requirement not met', 'does not match',
    'missing required', 'critical bug', 'data corruption',
    'logic error', 'incorrect behavior'
]

# MEDIUM: Performance issues, missing tests, code quality
MEDIUM_KEYWORDS = [
    'performance', 'slow', 'inefficient', 'optimization',
    'missing test', 'no tests', 'untested', 'test coverage',
    'code quality', 'maintainability', 'complexity',
    'duplication', 'refactor', 'improvement needed'
]

# LOW: Style issues, documentation, minor improvements
LOW_KEYWORDS = [
    'style', 'formatting', 'naming', 'documentation',
    'comment', 'typo', 'whitespace', 'minor',
    'suggestion', 'consider', 'could be better'
]

# All severity keywords organized by level
SEVERITY_KEYWORDS = {
    "critical": CRITICAL_KEYWORDS,
    "high": HIGH_KEYWORDS,
    "medium": MEDIUM_KEYWORDS,
    "low": LOW_KEYWORDS,
}


# =============================================================================
# PromptTemplate-based Prompts (New Format)
# =============================================================================


# Main fidelity review prompt - 6-section structure
FIDELITY_REVIEW_V1 = PromptTemplate(
    id="FIDELITY_REVIEW_V1",
    version="1.0",
    system_prompt="""You are an expert code reviewer performing implementation fidelity analysis.

Your role is to compare actual code implementation against specification requirements
and identify any deviations, issues, or concerns.

CRITICAL CONSTRAINTS:
- This is a READ-ONLY review - you MUST NOT write, create, or modify ANY files
- Execute code or commands - ANALYSIS ONLY
- Provide findings as structured JSON in your response

Focus on:
1. Requirement alignment - Does implementation match spec?
2. Success criteria - Are verification steps satisfied?
3. Deviations - Any divergences from specification?
4. Test coverage - Are tests comprehensive?
5. Code quality - Any maintainability concerns?
6. Documentation - Is implementation properly documented?""",
    user_template="""# Implementation Fidelity Review

## 1. Context
**Spec ID:** {spec_id}
**Spec Title:** {spec_title}
{spec_description}
**Review Scope:** {review_scope}

## 2. Specification Requirements
{spec_requirements}

## 3. Implementation Artifacts
{implementation_artifacts}

## 4. Test Results
{test_results}

## 5. Journal Entries
{journal_entries}

## 6. Review Questions

Please evaluate the implementation against the specification:

1. **Requirement Alignment:** Does the implementation match the spec requirements?
2. **Success Criteria:** Are all verification steps satisfied?
3. **Deviations:** Are there any deviations from the spec? If so, are they justified?
4. **Test Coverage:** Are tests comprehensive and passing?
5. **Code Quality:** Are there any quality, maintainability, or security concerns?
6. **Documentation:** Is the implementation properly documented?

### Required Response Format

Respond **only** with valid JSON matching the schema below. Do not include Markdown, prose, or additional commentary outside the JSON object.

```json
{response_schema}
```

Rules:
- Use lowercase values shown for enumerated fields (e.g., `verdict`, status flags)
- Keep arrays as arrays (use `[]` when a section has nothing to report)
- Populate `issues` and `recommendations` with key takeaways
- Feel free to include additional keys if needed, but never omit the ones above
- Severity levels for deviations: critical, high, medium, low""",
    required_context=[
        "spec_id",
        "spec_title",
        "review_scope",
        "spec_requirements",
        "implementation_artifacts",
    ],
    optional_context=[
        "spec_description",
        "test_results",
        "journal_entries",
        "response_schema",
    ],
    metadata={
        "workflow": "fidelity_review",
        "author": "system",
        "category": "implementation",
        "sections": [
            "Context",
            "Specification Requirements",
            "Implementation Artifacts",
            "Test Results",
            "Journal Entries",
            "Review Questions",
        ],
        "output_format": "json",
        "severity_levels": ["critical", "high", "medium", "low"],
    },
)


# Deviation analysis prompt - for deep-diving into identified deviations
FIDELITY_DEVIATION_ANALYSIS_V1 = PromptTemplate(
    id="FIDELITY_DEVIATION_ANALYSIS_V1",
    version="1.0",
    system_prompt="""You are an expert software architect analyzing specification deviations.

Your role is to assess the impact of identified deviations between implementation
and specification, determine risks, and recommend remediation strategies.

Focus on:
- Impact assessment (functional, security, performance, maintenance)
- Risk analysis and downstream effects
- Remediation options with effort/risk tradeoffs
- Prioritized recommendations""",
    user_template="""# Deviation Analysis

## Context
**Spec ID:** {spec_id}
**Analysis Scope:** {scope}

## Identified Deviations
{deviations_json}

## Original Requirements
{original_requirements}

## Analysis Requirements

For each deviation, provide:
1. **Impact Assessment**: Functional, security, performance, maintenance impact
2. **Risk Analysis**: Risks introduced by this deviation
3. **Downstream Effects**: How does this affect dependent components?
4. **Remediation Options**: Ways to address with effort/risk tradeoffs
5. **Recommendation**: accept|fix_now|fix_later|needs_discussion

### Required Response Format

Respond with valid JSON:

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
```""",
    required_context=["spec_id", "scope", "deviations_json", "original_requirements"],
    optional_context=[],
    metadata={
        "workflow": "fidelity_review",
        "author": "system",
        "category": "analysis",
        "output_format": "json",
    },
)


# Compliance summary prompt - for generating overall compliance reports
FIDELITY_COMPLIANCE_SUMMARY_V1 = PromptTemplate(
    id="FIDELITY_COMPLIANCE_SUMMARY_V1",
    version="1.0",
    system_prompt="""You are an expert technical lead generating compliance reports.

Your role is to synthesize fidelity review findings into an executive summary
with clear compliance status, prioritized issues, and sign-off recommendations.

Focus on:
- Overall compliance score and status
- Phase-by-phase breakdown
- Critical blocking issues
- Sign-off readiness assessment""",
    user_template="""# Compliance Summary Report

## Specification
**Spec ID:** {spec_id}
**Title:** {spec_title}
**Total Phases:** {total_phases}
**Total Tasks:** {total_tasks}

## Fidelity Review Data
{review_data}

## Summary Requirements

Generate a compliance summary addressing:
1. **Overall Compliance**: What is the overall compliance level?
2. **Phase Breakdown**: Compliance by phase
3. **Critical Issues**: List critical compliance issues
4. **Recommendations**: Prioritized recommendations
5. **Sign-off Status**: Is the implementation ready for approval?

### Required Response Format

Respond with valid JSON:

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
```""",
    required_context=["spec_id", "spec_title", "total_phases", "total_tasks", "review_data"],
    optional_context=[],
    metadata={
        "workflow": "fidelity_review",
        "author": "system",
        "category": "reporting",
        "output_format": "json",
    },
)


# =============================================================================
# Template Registry (PromptTemplate-based)
# =============================================================================


FIDELITY_REVIEW_TEMPLATES: Dict[str, PromptTemplate] = {
    "FIDELITY_REVIEW_V1": FIDELITY_REVIEW_V1,
    "FIDELITY_DEVIATION_ANALYSIS_V1": FIDELITY_DEVIATION_ANALYSIS_V1,
    "FIDELITY_COMPLIANCE_SUMMARY_V1": FIDELITY_COMPLIANCE_SUMMARY_V1,
}


# =============================================================================
# Legacy Prompt Templates (String-based for backward compatibility)
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
# Legacy Template Registry
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

    Supports both new PromptTemplate-based prompts (FIDELITY_*_V1) and
    legacy string templates for backward compatibility.
    """

    def __init__(self) -> None:
        """Initialize the builder with template registries."""
        self._templates = TEMPLATES
        self._prompt_templates = FIDELITY_REVIEW_TEMPLATES

    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a fidelity review prompt.

        Args:
            prompt_id: Template identifier. Supports:
                - PromptTemplate IDs: FIDELITY_REVIEW_V1, FIDELITY_DEVIATION_ANALYSIS_V1,
                  FIDELITY_COMPLIANCE_SUMMARY_V1
                - Legacy IDs: review_task, review_phase, compare_files,
                  deviation_analysis, compliance_summary
            context: Template context variables

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt_id is not recognized
        """
        # Check PromptTemplate registry first
        if prompt_id in self._prompt_templates:
            template = self._prompt_templates[prompt_id]

            # Provide defaults for optional context
            render_context = dict(context)

            # Add response schema default
            if "response_schema" not in render_context:
                render_context["response_schema"] = FIDELITY_RESPONSE_SCHEMA

            # Add empty defaults for optional fields
            if "spec_description" not in render_context:
                render_context["spec_description"] = ""
            if "test_results" not in render_context:
                render_context["test_results"] = "*No test results available*"
            if "journal_entries" not in render_context:
                render_context["journal_entries"] = "*No journal entries found*"

            return template.render(render_context)

        # Fall back to legacy templates
        template = self._templates.get(prompt_id)
        if template is None:
            all_ids = sorted(
                list(self._prompt_templates.keys()) + list(self._templates.keys())
            )
            available = ", ".join(all_ids)
            raise ValueError(
                f"Unknown prompt_id '{prompt_id}'. Available: {available}"
            )

        # Handle JSON content specially for legacy templates
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

    def list_prompts(self) -> List[str]:
        """Return available prompt IDs for fidelity review."""
        all_ids = list(self._prompt_templates.keys()) + list(self._templates.keys())
        return sorted(all_ids)

    def get_severity_keywords(self, level: str) -> List[str]:
        """
        Get severity categorization keywords for a given level.

        Args:
            level: Severity level (critical, high, medium, low)

        Returns:
            List of keywords for that severity level
        """
        return SEVERITY_KEYWORDS.get(level.lower(), [])


__all__ = [
    # PromptTemplate instances
    "FIDELITY_REVIEW_V1",
    "FIDELITY_DEVIATION_ANALYSIS_V1",
    "FIDELITY_COMPLIANCE_SUMMARY_V1",
    # Template registries
    "FIDELITY_REVIEW_TEMPLATES",
    "TEMPLATES",
    # Response schema
    "FIDELITY_RESPONSE_SCHEMA",
    # Severity keywords
    "SEVERITY_KEYWORDS",
    "CRITICAL_KEYWORDS",
    "HIGH_KEYWORDS",
    "MEDIUM_KEYWORDS",
    "LOW_KEYWORDS",
    # Builder
    "FidelityReviewPromptBuilder",
]
