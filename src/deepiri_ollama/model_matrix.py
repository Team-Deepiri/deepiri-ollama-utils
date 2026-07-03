"""Curated Ollama model fit matrix for the current host.

Read-only: scores every curated model id against detected RAM/VRAM tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deepiri_gpu_utils.detect import detect
from deepiri_gpu_utils.hardware import effective_vram_gb
from deepiri_gpu_utils.system_info import system_ram_gb

from .tiers import ModelFit, categorize_model, curated_models, setup_tier


@dataclass(frozen=True)
class ModelMatrixRow:
    """One model row in the fit matrix."""

    model: str
    description: str
    fit: ModelFit
    suitable: bool


@dataclass(frozen=True)
class ModelMatrix:
    """Full curated model fit matrix for this host."""

    setup_tier: str
    system_ram_gb: int
    effective_vram_gb: int
    backend: str
    rows: list[ModelMatrixRow] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def model_fit_matrix() -> ModelMatrix:
    """Return fit results for every curated Ollama model."""

    d = detect()
    ram = system_ram_gb()
    vram = effective_vram_gb(d, ram)
    tier = setup_tier(ram, vram)

    rows: list[ModelMatrixRow] = []
    counts: dict[str, int] = {"recommended": 0, "usable": 0, "marginal": 0, "no": 0}

    for model_id, desc in curated_models():
        fit = categorize_model(model_id, ram, vram)
        counts[fit] = counts.get(fit, 0) + 1
        rows.append(
            ModelMatrixRow(
                model=model_id,
                description=desc,
                fit=fit,
                suitable=fit in ("recommended", "usable"),
            )
        )

    return ModelMatrix(
        setup_tier=tier,
        system_ram_gb=ram,
        effective_vram_gb=vram,
        backend=d.backend,
        rows=rows,
        counts=counts,
    )


def render_model_matrix_text(matrix: ModelMatrix | None = None) -> str:
    """Render the model matrix as a fixed-width terminal table."""

    m = matrix or model_fit_matrix()
    lines = [
        f"model matrix (tier={m.setup_tier} ram={m.system_ram_gb}GB vram={m.effective_vram_gb}GB)",
        f"{'MODEL':<28} {'FIT':<12} DESCRIPTION",
        "-" * 72,
    ]
    for row in m.rows:
        lines.append(f"{row.model:<28} {row.fit:<12} {row.description[:40]}")
    lines.append("-" * 72)
    lines.append(
        "counts: "
        + " ".join(f"{k}={v}" for k, v in sorted(m.counts.items()))
    )
    return "\n".join(lines)
