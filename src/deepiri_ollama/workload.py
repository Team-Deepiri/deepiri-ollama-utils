"""Workload memory estimator for LLM inference on detected hardware.

Read-only heuristic: estimates whether a workload fits available RAM/VRAM using
simple parameter-count rules. Does not load models or touch GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deepiri_gpu_utils.detect import detect
from deepiri_gpu_utils.hardware import effective_vram_gb
from deepiri_gpu_utils.inventory import gpu_inventory
from deepiri_gpu_utils.system_info import system_ram_gb

# Approximate parameter counts (billions) for common families.
_MODEL_PARAMS_B: dict[str, float] = {
    "llama3.2:1b": 1.0,
    "llama3.2:3b": 3.0,
    "phi3:mini": 3.8,
    "gemma2:2b": 2.0,
    "mistral:7b": 7.0,
    "llama3:8b": 8.0,
    "llama3.1:8b": 8.0,
    "gemma2:9b": 9.0,
    "mistral-nemo:12b": 12.0,
    "codellama:13b": 13.0,
    "llama3.1:70b": 70.0,
    "mixtral:8x7b": 47.0,
    "gemma2:27b": 27.0,
}

_DEFAULT_PARAMS_B = 7.0
_BYTES_PER_PARAM_FP16 = 2
_OVERHEAD_FACTOR = 1.25


@dataclass(frozen=True)
class WorkloadEstimate:
    """Estimated memory footprint and fit verdict for a workload."""

    model: str
    parameters_b: float
    estimated_memory_gb: float
    available_memory_gb: float
    memory_source: str
    fits: bool
    headroom_gb: float
    notes: list[str] = field(default_factory=list)


def _params_for_model(model_name: str) -> float:
    key = model_name.strip().lower()
    if key in _MODEL_PARAMS_B:
        return _MODEL_PARAMS_B[key]
    if ":" in key:
        tag = key.split(":", 1)[1]
        for suffix, mult in (("70b", 70.0), ("27b", 27.0), ("13b", 13.0), ("12b", 12.0),
                             ("9b", 9.0), ("8b", 8.0), ("7b", 7.0), ("3b", 3.0), ("1b", 1.0)):
            if tag.endswith(suffix):
                return mult
    return _DEFAULT_PARAMS_B


def _available_memory_gb() -> tuple[float, str]:
    inv = gpu_inventory()
    free_values: list[float] = []
    total_values: list[float] = []
    for gpu in inv.gpus:
        if isinstance(gpu.details, dict):
            free = gpu.details.get("memory_free_gb")
            if isinstance(free, (int, float)):
                free_values.append(float(free))
        if gpu.memory_gb is not None:
            total_values.append(float(gpu.memory_gb))

    if free_values:
        return max(free_values), "gpu_free_vram"
    if total_values:
        return max(total_values), "gpu_total_vram"

    d = detect()
    ram = float(system_ram_gb())
    vram = float(effective_vram_gb(d, int(ram)))
    if vram > 0:
        return vram, "detect_vram"
    return ram, "system_ram"


def estimate_workload(model_name: str, *, context_tokens: int = 4096) -> WorkloadEstimate:
    """Estimate whether ``model_name`` fits available memory on this host."""

    params_b = _params_for_model(model_name)
    base_gb = params_b * _BYTES_PER_PARAM_FP16 * _OVERHEAD_FACTOR
    # Small context overhead heuristic (~0.5 GB per 4k tokens at 7B scale).
    context_gb = (context_tokens / 4096.0) * 0.5 * (params_b / 7.0)
    estimated = round(base_gb + context_gb, 2)

    available, source = _available_memory_gb()
    headroom = round(available - estimated, 2)
    fits = headroom >= 0

    notes: list[str] = [
        "Heuristic only: actual Ollama/PyTorch usage varies by quantisation and batch size.",
    ]
    if source == "system_ram":
        notes.append("No GPU VRAM reported; comparing against system RAM.")

    return WorkloadEstimate(
        model=model_name.strip(),
        parameters_b=params_b,
        estimated_memory_gb=estimated,
        available_memory_gb=round(available, 2),
        memory_source=source,
        fits=fits,
        headroom_gb=headroom,
        notes=notes,
    )
