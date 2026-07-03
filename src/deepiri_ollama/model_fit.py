"""Check whether a specific Ollama model fits the current hardware tier.

Read-only: reuses :mod:`tiers` and :mod:`deepiri_gpu_utils` detection without
pulling models or touching Docker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from deepiri_gpu_utils.detect import detect
from deepiri_gpu_utils.hardware import apply_backend_hint_vram, effective_vram_gb
from deepiri_gpu_utils.system_info import system_ram_gb

from .tiers import ModelFit, categorize_model, recommend_models, setup_tier

FitCategory = Literal["recommended", "usable", "marginal", "no"]


@dataclass(frozen=True)
class ModelFitResult:
    """Outcome of a single-model hardware fit check."""

    model: str
    fit: FitCategory
    setup_tier: str
    system_ram_gb: int
    effective_vram_gb: int
    default_model: str
    suitable: bool
    reason: str
    notes: list[str] = field(default_factory=list)


def model_fit_check(model_name: str, *, backend_hint: str | None = None) -> ModelFitResult:
    """Report whether ``model_name`` fits the detected hardware tier."""

    model = model_name.strip()
    d = detect()
    ram = system_ram_gb()
    vram, notes = apply_backend_hint_vram(effective_vram_gb(d, ram), ram, backend_hint)

    tier = setup_tier(ram, vram)
    fit: ModelFit = categorize_model(model, ram, vram)
    rec = recommend_models(backend_hint=backend_hint)

    suitable = fit in ("recommended", "usable")
    if fit == "recommended":
        reason = f"{model!r} is recommended for setup_tier={tier}."
    elif fit == "usable":
        reason = f"{model!r} is usable on this host (setup_tier={tier})."
    elif fit == "marginal":
        reason = f"{model!r} is marginal; expect slow or unstable inference."
    else:
        reason = f"{model!r} is not suitable for this host (setup_tier={tier})."

    return ModelFitResult(
        model=model,
        fit=fit,
        setup_tier=tier,
        system_ram_gb=ram,
        effective_vram_gb=vram,
        default_model=rec.default_model,
        suitable=suitable,
        reason=reason,
        notes=list(notes),
    )
