"""
Unit tests for AI consultation layer.

Tests the core consultation infrastructure including:
- ConsultationWorkflow enum
- ConsultationConfig dataclass
- PromptTemplate and PromptRegistry
- Workflow-specific prompt builders (doc_generation, plan_review, fidelity_review)
- AI error response helpers
"""

import pytest
from typing import Dict, Any

# Core consultation module
from foundry_mcp.core.ai_consultation import (
    ConsultationWorkflow,
    ConsultationRequest,
    ConsultationResult,
    ConsultationOrchestrator,
    ResultCache,
)

# Prompts infrastructure
from foundry_mcp.core.prompts import (
    PromptTemplate,
    PromptRegistry,
    PromptBuilder,
    get_prompt_builder,
    get_global_registry,
    reset_global_registry,
)

# Workflow-specific prompt builders
from foundry_mcp.core.prompts.doc_generation import (
    DOC_GEN_PROJECT_OVERVIEW_V1,
    DOC_GEN_ARCHITECTURE_V1,
    DOC_GEN_COMPONENT_INVENTORY_V1,
    DOC_GEN_TEMPLATES,
    DocGenerationPromptBuilder,
)

from foundry_mcp.core.prompts.plan_review import (
    PLAN_REVIEW_FULL_V1,
    PLAN_REVIEW_QUICK_V1,
    PLAN_REVIEW_SECURITY_V1,
    PLAN_REVIEW_FEASIBILITY_V1,
    SYNTHESIS_PROMPT_V1,
    PLAN_REVIEW_TEMPLATES,
    PlanReviewPromptBuilder,
)

from foundry_mcp.core.prompts.fidelity_review import (
    FIDELITY_REVIEW_V1,
    FIDELITY_DEVIATION_ANALYSIS_V1,
    FIDELITY_COMPLIANCE_SUMMARY_V1,
    FIDELITY_REVIEW_TEMPLATES,
    FIDELITY_RESPONSE_SCHEMA,
    SEVERITY_KEYWORDS,
    FidelityReviewPromptBuilder,
)

# Response helpers
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    ai_no_provider_error,
    ai_provider_timeout_error,
    ai_provider_error,
    ai_context_too_large_error,
    ai_prompt_not_found_error,
    ai_cache_stale_error,
)


# =============================================================================
# ConsultationWorkflow Tests
# =============================================================================


class TestConsultationWorkflow:
    """Tests for ConsultationWorkflow enum."""

    def test_workflow_values(self):
        """Workflow enum has expected values."""
        assert ConsultationWorkflow.DOC_GENERATION.value == "doc_generation"
        assert ConsultationWorkflow.PLAN_REVIEW.value == "plan_review"
        assert ConsultationWorkflow.FIDELITY_REVIEW.value == "fidelity_review"

    def test_workflow_count(self):
        """Verify all expected workflows are defined."""
        workflows = list(ConsultationWorkflow)
        assert len(workflows) == 3


# =============================================================================
# ConsultationRequest Tests
# =============================================================================


class TestConsultationRequest:
    """Tests for ConsultationRequest dataclass."""

    def test_request_creation(self):
        """Request can be created with valid parameters."""
        request = ConsultationRequest(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            prompt_id="PLAN_REVIEW_FULL_V1",
            context={"spec_id": "test-001"},
        )
        assert request.workflow == ConsultationWorkflow.PLAN_REVIEW
        assert request.prompt_id == "PLAN_REVIEW_FULL_V1"
        assert request.context["spec_id"] == "test-001"

    def test_request_defaults(self):
        """Request has expected defaults."""
        request = ConsultationRequest(
            workflow=ConsultationWorkflow.DOC_GENERATION,
            prompt_id="test",
            context={},
        )
        assert request.provider_id is None
        assert request.timeout == 120.0  # Default timeout
        assert request.model is None


# =============================================================================
# ConsultationResult Tests
# =============================================================================


class TestConsultationResult:
    """Tests for ConsultationResult dataclass."""

    def test_result_success(self):
        """Result can represent successful consultation."""
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Generated content",
            provider_id="gemini",
            model_used="gemini-2.0-flash",
        )
        assert result.content == "Generated content"
        assert result.provider_id == "gemini"
        assert result.model_used == "gemini-2.0-flash"
        assert result.error is None

    def test_result_with_error(self):
        """Result can include error information."""
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="",
            provider_id="gemini",
            model_used="gemini-2.0-flash",
            error="Provider timeout",
        )
        assert result.error == "Provider timeout"


# =============================================================================
# PromptTemplate Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_template_creation(self):
        """Template can be created with valid parameters."""
        template = PromptTemplate(
            id="test_template",
            version="1.0",
            system_prompt="System message",
            user_template="User message: {content}",
            required_context=["content"],
        )
        assert template.id == "test_template"
        assert template.version == "1.0"

    def test_template_validation_empty_id(self):
        """Template rejects empty ID."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            PromptTemplate(
                id="",
                version="1.0",
                system_prompt="System",
                user_template="User",
            )

    def test_template_get_variables(self):
        """Template extracts variables from user_template."""
        template = PromptTemplate(
            id="test",
            version="1.0",
            system_prompt="System",
            user_template="Hello {name}, your score is {score}",
            required_context=["name", "score"],
        )
        variables = template.get_variables()
        assert "name" in variables
        assert "score" in variables

    def test_template_validate_context(self):
        """Template validates context contains required keys."""
        template = PromptTemplate(
            id="test",
            version="1.0",
            system_prompt="System",
            user_template="{required_key}",
            required_context=["required_key"],
        )
        # Valid context
        missing = template.validate_context({"required_key": "value"})
        assert missing == []

        # Missing context
        missing = template.validate_context({})
        assert "required_key" in missing

    def test_template_render(self):
        """Template renders with context substitution."""
        template = PromptTemplate(
            id="test",
            version="1.0",
            system_prompt="System",
            user_template="Hello {name}!",
            required_context=["name"],
        )
        result = template.render({"name": "World"})
        assert result == "Hello World!"


# =============================================================================
# PromptRegistry Tests
# =============================================================================


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_registry_register_and_get(self):
        """Registry can register and retrieve templates."""
        registry = PromptRegistry()
        template = PromptTemplate(
            id="test_template",
            version="1.0",
            system_prompt="System",
            user_template="User",
        )
        registry.register(template)

        retrieved = registry.get("test_template")
        assert retrieved is not None
        assert retrieved.id == "test_template"

    def test_registry_duplicate_registration(self):
        """Registry rejects duplicate IDs by default."""
        registry = PromptRegistry()
        template = PromptTemplate(
            id="test",
            version="1.0",
            system_prompt="System",
            user_template="User",
        )
        registry.register(template)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(template)

    def test_registry_list_templates(self):
        """Registry lists all registered template IDs."""
        registry = PromptRegistry()
        registry.register(PromptTemplate(id="a", version="1.0", system_prompt="S", user_template="U"))
        registry.register(PromptTemplate(id="b", version="1.0", system_prompt="S", user_template="U"))

        templates = registry.list_templates()
        assert "a" in templates
        assert "b" in templates


# =============================================================================
# Prompt Builder Factory Tests
# =============================================================================


class TestPromptBuilderFactory:
    """Tests for get_prompt_builder factory."""

    def test_factory_returns_doc_generation_builder(self):
        """Factory returns DocGenerationPromptBuilder for DOC_GENERATION."""
        builder = get_prompt_builder(ConsultationWorkflow.DOC_GENERATION)
        assert isinstance(builder, DocGenerationPromptBuilder)

    def test_factory_returns_plan_review_builder(self):
        """Factory returns PlanReviewPromptBuilder for PLAN_REVIEW."""
        builder = get_prompt_builder(ConsultationWorkflow.PLAN_REVIEW)
        assert isinstance(builder, PlanReviewPromptBuilder)

    def test_factory_returns_fidelity_review_builder(self):
        """Factory returns FidelityReviewPromptBuilder for FIDELITY_REVIEW."""
        builder = get_prompt_builder(ConsultationWorkflow.FIDELITY_REVIEW)
        assert isinstance(builder, FidelityReviewPromptBuilder)


# =============================================================================
# DocGenerationPromptBuilder Tests
# =============================================================================


class TestDocGenerationPromptBuilder:
    """Tests for DocGenerationPromptBuilder."""

    def test_builder_list_prompts(self):
        """Builder lists available prompts."""
        builder = DocGenerationPromptBuilder()
        prompts = builder.list_prompts()

        # Should include both new and legacy prompts
        assert "DOC_GEN_PROJECT_OVERVIEW_V1" in prompts
        assert "DOC_GEN_ARCHITECTURE_V1" in prompts
        assert "DOC_GEN_COMPONENT_INVENTORY_V1" in prompts

    def test_builder_build_overview_prompt(self):
        """Builder renders DOC_GEN_PROJECT_OVERVIEW_V1."""
        builder = DocGenerationPromptBuilder()
        context = {
            "project_context": "A test project for demonstration",
            "key_files": "main.py, utils.py",
        }
        result = builder.build("DOC_GEN_PROJECT_OVERVIEW_V1", context)

        # Verify context was injected
        assert "A test project for demonstration" in result
        assert "main.py, utils.py" in result
        # Verify template structure
        assert "Project Overview Research" in result
        assert "Research Findings" in result

    def test_templates_have_metadata(self):
        """DOC_GEN templates have expected metadata."""
        # Check workflow is present in metadata
        assert "workflow" in DOC_GEN_PROJECT_OVERVIEW_V1.metadata
        assert "workflow" in DOC_GEN_ARCHITECTURE_V1.metadata


# =============================================================================
# PlanReviewPromptBuilder Tests
# =============================================================================


class TestPlanReviewPromptBuilder:
    """Tests for PlanReviewPromptBuilder."""

    def test_builder_list_prompts(self):
        """Builder lists available prompts."""
        builder = PlanReviewPromptBuilder()
        prompts = builder.list_prompts()

        assert "PLAN_REVIEW_FULL_V1" in prompts
        assert "PLAN_REVIEW_QUICK_V1" in prompts
        assert "PLAN_REVIEW_SECURITY_V1" in prompts
        assert "PLAN_REVIEW_FEASIBILITY_V1" in prompts
        assert "SYNTHESIS_PROMPT_V1" in prompts

    def test_builder_build_full_review(self):
        """Builder renders PLAN_REVIEW_FULL_V1."""
        builder = PlanReviewPromptBuilder()
        context = {
            "spec_id": "test-spec-001",
            "title": "Test Specification",
            "version": "1.0",
            "spec_content": "Test content here",
        }
        result = builder.build("PLAN_REVIEW_FULL_V1", context)

        assert "test-spec-001" in result
        assert "Test Specification" in result

    def test_full_review_has_six_dimensions(self):
        """PLAN_REVIEW_FULL_V1 metadata includes 6 dimensions."""
        dimensions = PLAN_REVIEW_FULL_V1.metadata.get("dimensions", [])
        assert len(dimensions) == 6
        assert "Completeness" in dimensions
        assert "Security" in dimensions

    def test_security_review_has_metadata(self):
        """PLAN_REVIEW_SECURITY_V1 has workflow metadata."""
        assert "workflow" in PLAN_REVIEW_SECURITY_V1.metadata
        assert PLAN_REVIEW_SECURITY_V1.id == "PLAN_REVIEW_SECURITY_V1"


# =============================================================================
# FidelityReviewPromptBuilder Tests
# =============================================================================


class TestFidelityReviewPromptBuilder:
    """Tests for FidelityReviewPromptBuilder."""

    def test_builder_list_prompts(self):
        """Builder lists available prompts."""
        builder = FidelityReviewPromptBuilder()
        prompts = builder.list_prompts()

        assert "FIDELITY_REVIEW_V1" in prompts
        assert "FIDELITY_DEVIATION_ANALYSIS_V1" in prompts
        assert "FIDELITY_COMPLIANCE_SUMMARY_V1" in prompts
        # Legacy prompts
        assert "review_task" in prompts
        assert "review_phase" in prompts

    def test_builder_build_fidelity_review(self):
        """Builder renders FIDELITY_REVIEW_V1."""
        builder = FidelityReviewPromptBuilder()
        context = {
            "spec_id": "test-spec-001",
            "spec_title": "Test Spec",
            "review_scope": "task-1-1",
            "spec_requirements": "Implement feature X",
            "implementation_artifacts": "git diff here",
        }
        result = builder.build("FIDELITY_REVIEW_V1", context)

        assert "test-spec-001" in result
        assert "Implementation Fidelity Review" in result

    def test_fidelity_has_six_sections(self):
        """FIDELITY_REVIEW_V1 metadata includes 6 sections."""
        sections = FIDELITY_REVIEW_V1.metadata.get("sections", [])
        assert len(sections) == 6
        assert "Context" in sections
        assert "Review Questions" in sections

    def test_severity_keywords_defined(self):
        """Severity keywords are defined for all levels."""
        assert "critical" in SEVERITY_KEYWORDS
        assert "high" in SEVERITY_KEYWORDS
        assert "medium" in SEVERITY_KEYWORDS
        assert "low" in SEVERITY_KEYWORDS

    def test_get_severity_keywords(self):
        """Builder returns severity keywords for a level."""
        builder = FidelityReviewPromptBuilder()
        critical_kw = builder.get_severity_keywords("critical")
        assert "security" in critical_kw

    def test_response_schema_structure(self):
        """Fidelity response schema has expected fields."""
        assert "verdict" in FIDELITY_RESPONSE_SCHEMA
        assert "deviations" in FIDELITY_RESPONSE_SCHEMA
        assert "recommendations" in FIDELITY_RESPONSE_SCHEMA


# =============================================================================
# AI Error Response Tests
# =============================================================================


class TestAIErrorCodes:
    """Tests for AI-specific error codes."""

    def test_error_codes_defined(self):
        """AI error codes are defined in ErrorCode enum."""
        assert ErrorCode.AI_NO_PROVIDER == "AI_NO_PROVIDER"
        assert ErrorCode.AI_PROVIDER_TIMEOUT == "AI_PROVIDER_TIMEOUT"
        assert ErrorCode.AI_PROVIDER_ERROR == "AI_PROVIDER_ERROR"
        assert ErrorCode.AI_CONTEXT_TOO_LARGE == "AI_CONTEXT_TOO_LARGE"
        assert ErrorCode.AI_PROMPT_NOT_FOUND == "AI_PROMPT_NOT_FOUND"
        assert ErrorCode.AI_CACHE_STALE == "AI_CACHE_STALE"

    def test_error_type_ai_provider(self):
        """AI_PROVIDER error type is defined."""
        assert ErrorType.AI_PROVIDER == "ai_provider"


class TestAIErrorHelpers:
    """Tests for AI error response helpers."""

    def test_ai_no_provider_error(self):
        """ai_no_provider_error creates correct response."""
        resp = ai_no_provider_error(
            "No AI provider available",
            required_providers=["gemini", "codex"],
        )
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_NO_PROVIDER
        assert resp.data["required_providers"] == ["gemini", "codex"]
        assert "remediation" in resp.data

    def test_ai_provider_timeout_error(self):
        """ai_provider_timeout_error creates correct response."""
        resp = ai_provider_timeout_error("gemini", 300)
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_PROVIDER_TIMEOUT
        assert resp.data["provider_id"] == "gemini"
        assert resp.data["timeout_seconds"] == 300

    def test_ai_provider_error(self):
        """ai_provider_error creates correct response."""
        resp = ai_provider_error("gemini", "Invalid API key", status_code=401)
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_PROVIDER_ERROR
        assert resp.data["status_code"] == 401

    def test_ai_context_too_large_error(self):
        """ai_context_too_large_error creates correct response."""
        resp = ai_context_too_large_error(150000, 128000, provider_id="gemini")
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_CONTEXT_TOO_LARGE
        assert resp.data["overflow_tokens"] == 22000

    def test_ai_prompt_not_found_error(self):
        """ai_prompt_not_found_error creates correct response."""
        resp = ai_prompt_not_found_error(
            "INVALID_PROMPT",
            available_prompts=["PLAN_REVIEW_FULL_V1"],
            workflow="plan_review",
        )
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_PROMPT_NOT_FOUND
        assert resp.data["workflow"] == "plan_review"

    def test_ai_cache_stale_error(self):
        """ai_cache_stale_error creates correct response."""
        resp = ai_cache_stale_error("key", 7200, 3600)
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_CACHE_STALE
        assert resp.data["cache_age_seconds"] == 7200

    def test_all_helpers_include_remediation(self):
        """All AI error helpers include remediation guidance."""
        responses = [
            ai_no_provider_error(),
            ai_provider_timeout_error("test", 60),
            ai_provider_error("test", "error"),
            ai_context_too_large_error(100, 50),
            ai_prompt_not_found_error("test"),
            ai_cache_stale_error("key", 100, 50),
        ]
        for resp in responses:
            assert "remediation" in resp.data
            assert resp.data["remediation"]  # Not empty
