"""Review helpers for the unified `review(action=...)` router.

This module contains the reusable review execution functions (quick structural
review + AI-backed plan review). It intentionally does not register any
standalone MCP tools.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.responses import error_response, success_response

logger = logging.getLogger(__name__)
_metrics = get_metrics()

REVIEW_TYPES = ["quick", "full", "security", "feasibility"]
REVIEW_TYPE_TO_TEMPLATE = {
    "full": "PLAN_REVIEW_FULL_V1",
    "security": "PLAN_REVIEW_SECURITY_V1",
    "feasibility": "PLAN_REVIEW_FEASIBILITY_V1",
}

DEFAULT_AI_TIMEOUT = 120.0


def _get_llm_status() -> Dict[str, Any]:
    """Return basic LLM configuration status."""

    try:
        from foundry_mcp.core.llm_config import get_llm_config

        config = get_llm_config()
        return {
            "configured": config.get_api_key() is not None,
            "provider": config.provider.value,
            "model": config.get_model(),
        }
    except ImportError:
        return {"configured": False, "error": "LLM config not available"}
    except Exception as exc:
        logger.debug("Failed to get LLM config: %s", exc)
        return {"configured": False, "error": "Failed to load LLM configuration"}


def _run_quick_review(
    *,
    spec_id: str,
    path: Optional[str],
    dry_run: bool,
    llm_status: Dict[str, Any],
    start_time: float,
) -> dict:
    """Run a quick (non-LLM) structural review."""

    from foundry_mcp.core.review import quick_review
    from foundry_mcp.core.spec import find_specs_directory

    specs_dir = find_specs_directory(path) if path else find_specs_directory()

    if dry_run:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return asdict(
            success_response(
                spec_id=spec_id,
                review_type="quick",
                dry_run=True,
                llm_status=llm_status,
                message="Dry run - quick review skipped",
                duration_ms=round(duration_ms, 2),
            )
        )

    result = quick_review(spec_id=spec_id, specs_dir=specs_dir)
    duration_ms = (time.perf_counter() - start_time) * 1000

    findings = [
        {
            "code": finding.code,
            "message": finding.message,
            "severity": finding.severity,
            "category": finding.category,
            "location": finding.location,
            "suggestion": finding.suggestion,
        }
        for finding in result.findings
    ]

    return asdict(
        success_response(
            spec_id=result.spec_id,
            title=result.title,
            review_type=result.review_type,
            is_valid=result.is_valid,
            findings=findings,
            summary=result.summary,
            error_count=result.error_count,
            warning_count=result.warning_count,
            info_count=result.info_count,
            llm_status=llm_status,
            duration_ms=round(duration_ms, 2),
        )
    )


def _run_ai_review(
    *,
    spec_id: str,
    review_type: str,
    ai_provider: Optional[str],
    ai_timeout: float,
    consultation_cache: bool,
    path: Optional[str],
    dry_run: bool,
    llm_status: Dict[str, Any],
    start_time: float,
) -> dict:
    """Run an AI-backed plan review using ConsultationOrchestrator."""

    from foundry_mcp.core.review import prepare_review_context
    from foundry_mcp.core.spec import find_specs_directory

    specs_dir = find_specs_directory(path) if path else find_specs_directory()

    template_id = REVIEW_TYPE_TO_TEMPLATE.get(review_type)
    if template_id is None:
        return asdict(
            error_response(
                f"Unknown review type: {review_type}",
                error_code="INVALID_REVIEW_TYPE",
                error_type="validation",
                data={"review_type": review_type},
                remediation=f"Use one of: {', '.join(REVIEW_TYPE_TO_TEMPLATE.keys())}",
            )
        )

    context = prepare_review_context(
        spec_id=spec_id,
        specs_dir=specs_dir,
        include_tasks=True,
        include_journals=True,
    )

    if context is None:
        return asdict(
            error_response(
                f"Specification '{spec_id}' not found",
                error_code="SPEC_NOT_FOUND",
                error_type="not_found",
                data={"spec_id": spec_id},
                remediation="Verify the spec_id exists in specs/",
            )
        )

    if dry_run:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return asdict(
            success_response(
                spec_id=spec_id,
                review_type=review_type,
                template_id=template_id,
                dry_run=True,
                llm_status=llm_status,
                ai_provider=ai_provider,
                consultation_cache=consultation_cache,
                message=f"Dry run - {review_type} review would use template {template_id}",
                spec_title=context.title,
                task_count=context.stats.totals.get("tasks", 0) if context.stats else 0,
                duration_ms=round(duration_ms, 2),
            )
        )

    from foundry_mcp.core.ai_consultation import (
        ConsensusResult,
        ConsultationOrchestrator,
        ConsultationRequest,
        ConsultationWorkflow,
    )

    orchestrator = ConsultationOrchestrator(
        preferred_providers=[ai_provider] if ai_provider else [],
        default_timeout=ai_timeout,
    )

    if not orchestrator.is_available(provider_id=ai_provider):
        provider_msg = f" (requested: {ai_provider})" if ai_provider else ""
        _metrics.counter(
            "review.errors",
            labels={"tool": "review", "error_type": "ai_no_provider"},
        )
        return asdict(
            error_response(
                f"AI-enhanced review requested but no providers available{provider_msg}",
                error_code="AI_NO_PROVIDER",
                error_type="unavailable",
                data={
                    "spec_id": spec_id,
                    "review_type": review_type,
                    "requested_provider": ai_provider,
                    "llm_status": llm_status,
                },
                remediation="Install/configure an AI provider or use review_type='quick'",
            )
        )

    spec_content = json.dumps(context.spec_data, indent=2)
    request = ConsultationRequest(
        workflow=ConsultationWorkflow.PLAN_REVIEW,
        prompt_id=template_id,
        context={
            "spec_content": spec_content,
            "spec_id": spec_id,
            "title": context.title,
            "review_type": review_type,
        },
        provider_id=ai_provider,
        timeout=ai_timeout,
    )

    try:
        result = orchestrator.consult(request, use_cache=consultation_cache)
    except Exception as exc:
        logger.exception("AI consultation failed for %s", spec_id)
        return asdict(
            error_response(
                "AI consultation failed",
                error_code="AI_CONSULTATION_ERROR",
                error_type="error",
                data={"spec_id": spec_id, "review_type": review_type},
                remediation="Check provider configuration and try again",
            )
        )

    duration_ms = (time.perf_counter() - start_time) * 1000

    if isinstance(result, ConsensusResult):
        responses_data = [
            {
                "provider_id": response.provider_id,
                "model_used": response.model_used,
                "content": response.content,
                "success": response.success,
                "error": response.error,
                "tokens": response.tokens,
                "duration_ms": response.duration_ms,
            }
            for response in result.responses
        ]

        agreement_data = None
        if result.agreement:
            agreement_data = {
                "total_providers": result.agreement.total_providers,
                "successful_providers": result.agreement.successful_providers,
                "failed_providers": result.agreement.failed_providers,
                "success_rate": result.agreement.success_rate,
                "has_consensus": result.agreement.has_consensus,
            }

        return asdict(
            success_response(
                spec_id=spec_id,
                title=context.title,
                review_type=review_type,
                template_id=template_id,
                llm_status=llm_status,
                mode="multi_model",
                consultation_cache=consultation_cache,
                responses=responses_data,
                agreement=agreement_data,
                primary_content=result.primary_content,
                warnings=result.warnings,
                duration_ms=round(duration_ms, 2),
            )
        )

    return asdict(
        success_response(
            spec_id=spec_id,
            title=context.title,
            review_type=review_type,
            template_id=template_id,
            llm_status=llm_status,
            mode="single_model",
            ai_provider=getattr(result, "provider_id", None) if result else ai_provider,
            consultation_cache=consultation_cache,
            response=getattr(result, "content", None) if result else None,
            model=getattr(result, "model_used", None) if result else None,
            cached=getattr(result, "cache_hit", False) if result else False,
            duration_ms=round(duration_ms, 2),
        )
    )


__all__ = [
    "DEFAULT_AI_TIMEOUT",
    "REVIEW_TYPES",
    "_get_llm_status",
    "_run_ai_review",
    "_run_quick_review",
]
