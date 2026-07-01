"""GPU-aware Ollama model management built from deepiri-gpu-utils primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

from deepiri_gpu_utils.capacity import ModelCapacity, model_capacity
from deepiri_gpu_utils.model_fit import ModelFitResult, model_fit_check
from deepiri_gpu_utils.model_matrix import ModelMatrix, model_fit_matrix
from deepiri_gpu_utils.ollama import (
    categorize_model,
    curated_model_ids,
    curated_models,
    recommend_models,
    setup_tier,
)
from deepiri_gpu_utils.workload import WorkloadEstimate, estimate_workload

from .client import OllamaAPIError, OllamaClient, OllamaReadiness, PullResult


@dataclass(frozen=True)
class RecommendationRow:
    """One curated model enriched with fit and installation state."""

    model: str
    description: str
    fit: str
    suitable: bool
    installed: bool


@dataclass(frozen=True)
class RecommendationReport:
    """Hardware recommendations combined with local Ollama state."""

    default_model: str
    setup_tier: str
    system_ram_gb: int
    effective_vram_gb: int
    ready: bool
    base_url: str
    installed_models: list[str] = field(default_factory=list)
    rows: list[RecommendationRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    readiness_error: str | None = None


@dataclass(frozen=True)
class ModelAssessment:
    """Fit, workload, and capacity facts for a requested model."""

    model: str
    fit: ModelFitResult
    workload: WorkloadEstimate
    capacity: ModelCapacity
    warning: str | None = None


@dataclass(frozen=True)
class InstallResult:
    """Outcome of an Ollama pull followed by installed-model verification."""

    model: str
    success: bool
    already_installed: bool
    pulled: bool
    verified: bool
    assessment: ModelAssessment
    pull: PullResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class RemoveResult:
    """Outcome of deleting and then re-checking an Ollama model."""

    model: str
    success: bool
    existed: bool
    verified_absent: bool
    error: str | None = None


def fit_warning(result: ModelFitResult) -> str | None:
    """Translate gpu-utils' fit verdict into an operator-facing warning."""

    if result.fit == "recommended":
        return None
    if result.fit == "usable":
        return f"{result.model} is usable but may have limited headroom on this host."
    if result.fit == "marginal":
        return f"{result.model} is marginal; inference may be slow or unstable."
    return f"{result.model} is not suitable for this host: {result.reason}"


def assess_model(
    model_name: str,
    *,
    backend_hint: str | None = None,
    context_tokens: int = 4096,
    reserved_gb: float = 1.0,
) -> ModelAssessment:
    """Build an assessment entirely from deepiri-gpu-utils library APIs."""

    model = model_name.strip()
    if not model:
        raise ValueError("model_name must not be empty")

    fit = model_fit_check(model, backend_hint=backend_hint)
    workload = estimate_workload(model, context_tokens=context_tokens)
    capacity = model_capacity(
        model,
        reserved_gb=reserved_gb,
        context_tokens=context_tokens,
    )
    return ModelAssessment(
        model=model,
        fit=fit,
        workload=workload,
        capacity=capacity,
        warning=fit_warning(fit),
    )


def model_matrix() -> ModelMatrix:
    """Return gpu-utils' curated model matrix without reimplementing sizing logic."""

    return model_fit_matrix()


async def recommendation_report(
    client: OllamaClient,
    *,
    backend_hint: str | None = None,
) -> RecommendationReport:
    """Combine gpu-utils recommendations with the current Ollama inventory."""

    recommendation = recommend_models(backend_hint=backend_hint)
    readiness: OllamaReadiness = await client.readiness()
    installed = set(readiness.models)
    descriptions = dict(curated_models())

    rows: list[RecommendationRow] = []
    for model in curated_model_ids():
        fit = categorize_model(
            model,
            recommendation.system_ram_gb,
            recommendation.effective_vram_gb,
        )
        rows.append(
            RecommendationRow(
                model=model,
                description=descriptions.get(model, ""),
                fit=fit,
                suitable=fit in {"recommended", "usable"},
                installed=model in installed,
            )
        )

    tier = setup_tier(
        recommendation.system_ram_gb,
        recommendation.effective_vram_gb,
    )
    return RecommendationReport(
        default_model=recommendation.default_model,
        setup_tier=tier,
        system_ram_gb=recommendation.system_ram_gb,
        effective_vram_gb=recommendation.effective_vram_gb,
        ready=readiness.ready,
        base_url=readiness.base_url,
        installed_models=readiness.models,
        rows=rows,
        notes=list(recommendation.notes),
        readiness_error=readiness.error,
    )


async def install_model(
    client: OllamaClient,
    model_name: str,
    *,
    backend_hint: str | None = None,
    context_tokens: int = 4096,
    reserved_gb: float = 1.0,
) -> InstallResult:
    """Pull a model, retaining fit facts and verifying installation afterward."""

    assessment = assess_model(
        model_name,
        backend_hint=backend_hint,
        context_tokens=context_tokens,
        reserved_gb=reserved_gb,
    )
    model = assessment.model
    pulled = False

    try:
        if await client.has_model(model):
            return InstallResult(
                model=model,
                success=True,
                already_installed=True,
                pulled=False,
                verified=True,
                assessment=assessment,
            )

        pull = await client.pull_model(model)
        if not pull.success:
            return InstallResult(
                model=model,
                success=False,
                already_installed=False,
                pulled=False,
                verified=False,
                assessment=assessment,
                pull=pull,
                error=pull.error or "Ollama did not complete the model pull",
            )

        pulled = True
        verified = await client.has_model(model)
        return InstallResult(
            model=model,
            success=verified,
            already_installed=False,
            pulled=True,
            verified=verified,
            assessment=assessment,
            pull=pull,
            error=None if verified else "Model pull completed but verification failed",
        )
    except OllamaAPIError as exc:
        return InstallResult(
            model=model,
            success=False,
            already_installed=False,
            pulled=pulled,
            verified=False,
            assessment=assessment,
            error=str(exc),
        )


async def remove_model(client: OllamaClient, model_name: str) -> RemoveResult:
    """Delete a model and verify that it no longer appears in Ollama."""

    model = model_name.strip()
    if not model:
        raise ValueError("model_name must not be empty")

    existed = False
    try:
        existed = await client.has_model(model)
        if not existed:
            return RemoveResult(
                model=model,
                success=True,
                existed=False,
                verified_absent=True,
            )

        await client.delete_model(model)
        verified_absent = not await client.has_model(model)
        return RemoveResult(
            model=model,
            success=verified_absent,
            existed=True,
            verified_absent=verified_absent,
            error=None if verified_absent else "Delete completed but the model is still listed",
        )
    except OllamaAPIError as exc:
        return RemoveResult(
            model=model,
            success=False,
            existed=existed,
            verified_absent=False,
            error=str(exc),
        )
