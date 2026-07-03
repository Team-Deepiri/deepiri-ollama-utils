"""Estimate how many concurrent model instances fit available memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from .workload import estimate_workload


@dataclass(frozen=True)
class ModelCapacity:
    """How many concurrent instances of a model likely fit."""

    model: str
    per_instance_gb: float
    available_gb: float
    reserved_gb: float
    max_instances: int
    memory_source: str
    notes: list[str] = field(default_factory=list)


def model_capacity(
    model_name: str,
    *,
    reserved_gb: float = 1.0,
    context_tokens: int = 4096,
) -> ModelCapacity:
    """Estimate concurrent model slots from :func:`workload.estimate_workload`."""

    estimate = estimate_workload(model_name, context_tokens=context_tokens)
    usable = max(estimate.available_memory_gb - reserved_gb, 0.0)
    per = estimate.estimated_memory_gb
    if per <= 0:
        slots = 0
    else:
        slots = int(usable // per)

    notes = list(estimate.notes)
    notes.append(f"Reserved {reserved_gb} GB headroom for OS/runtime.")
    if slots == 0 and estimate.fits:
        slots = 1
        notes.append("At least one instance fits; concurrent slots rounded down to 0.")

    return ModelCapacity(
        model=estimate.model,
        per_instance_gb=estimate.estimated_memory_gb,
        available_gb=estimate.available_memory_gb,
        reserved_gb=reserved_gb,
        max_instances=max(0, slots),
        memory_source=estimate.memory_source,
        notes=notes,
    )
