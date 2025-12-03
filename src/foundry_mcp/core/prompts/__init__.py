"""
Prompt templates and builders for AI consultation workflows.

This package provides workflow-specific prompt builders that generate
structured prompts for different AI consultation use cases.

Workflow Coverage:
    - doc_generation: Generate documentation from code analysis
    - plan_review: Review and critique SDD specifications
    - fidelity_review: Compare implementation against specifications

Example Usage:
    from foundry_mcp.core.prompts import get_prompt_builder
    from foundry_mcp.core.ai_consultation import ConsultationWorkflow

    builder = get_prompt_builder(ConsultationWorkflow.DOC_GENERATION)
    prompt = builder.build("analyze_module", {"content": "def foo(): pass"})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from foundry_mcp.core.ai_consultation import ConsultationWorkflow


class PromptBuilder(ABC):
    """
    Abstract base class for workflow-specific prompt builders.

    Subclasses implement prompt templates for specific consultation workflows,
    providing a consistent interface for prompt generation.

    Methods:
        build: Generate a prompt from a template ID and context
        list_prompts: Return available prompt IDs for this workflow
    """

    @abstractmethod
    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a prompt from template ID and context.

        Args:
            prompt_id: Identifier for the prompt template
            context: Structured context data to inject

        Returns:
            The rendered prompt string

        Raises:
            ValueError: If prompt_id is not recognized
        """
        raise NotImplementedError

    @abstractmethod
    def list_prompts(self) -> list[str]:
        """
        Return list of available prompt IDs.

        Returns:
            List of prompt template IDs
        """
        raise NotImplementedError


def get_prompt_builder(workflow: "ConsultationWorkflow") -> PromptBuilder:
    """
    Get the prompt builder for a consultation workflow.

    Args:
        workflow: The consultation workflow type

    Returns:
        PromptBuilder instance for the workflow

    Raises:
        ValueError: If workflow is not supported
    """
    # Import here to avoid circular imports
    from foundry_mcp.core.ai_consultation import ConsultationWorkflow
    from foundry_mcp.core.prompts.doc_generation import DocGenerationPromptBuilder
    from foundry_mcp.core.prompts.plan_review import PlanReviewPromptBuilder
    from foundry_mcp.core.prompts.fidelity_review import FidelityReviewPromptBuilder

    builders: Dict[ConsultationWorkflow, type[PromptBuilder]] = {
        ConsultationWorkflow.DOC_GENERATION: DocGenerationPromptBuilder,
        ConsultationWorkflow.PLAN_REVIEW: PlanReviewPromptBuilder,
        ConsultationWorkflow.FIDELITY_REVIEW: FidelityReviewPromptBuilder,
    }

    builder_class = builders.get(workflow)
    if builder_class is None:
        raise ValueError(f"Unsupported workflow: {workflow}")

    return builder_class()


__all__ = [
    "PromptBuilder",
    "get_prompt_builder",
]
