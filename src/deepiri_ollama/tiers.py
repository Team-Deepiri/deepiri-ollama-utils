"""Ollama model tiers ported from diri-cyrex/scripts/llm/check-ollama-models.sh (logic only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from deepiri_gpu_utils.detect import detect
from deepiri_gpu_utils.hardware import apply_backend_hint_vram, effective_vram_gb
from deepiri_gpu_utils.system_info import system_ram_gb

ModelFit = Literal["recommended", "usable", "marginal", "no"]


@dataclass(frozen=True)
class OllamaRecommendation:
    default_model: str
    recommended_models: list[str]
    usable_models: list[str]
    marginal_models: list[str]
    unsuitable_models: list[str]
    notes: list[str] = field(default_factory=list)
    setup_tier: str = "unknown"
    system_ram_gb: int = 0
    effective_vram_gb: int = 0
    category: Literal["hardware_tiered"] = "hardware_tiered"


def setup_tier(ram_gb: int, vram_gb: int) -> str:
    """Mirror check-ollama-models.sh SETUP_CATEGORY rules (lines ~667–696)."""

    if (ram_gb >= 32 or ram_gb >= 30) and (vram_gb >= 16 or vram_gb >= 15):
        return "setup5"
    if ram_gb >= 32 and vram_gb >= 10:
        return "setup4"
    if ram_gb >= 32 and vram_gb >= 8:
        return "setup3"
    if vram_gb >= 15:
        return "setup5"
    if ram_gb >= 16 and vram_gb >= 10:
        return "setup2"
    if ram_gb >= 16 and vram_gb >= 8:
        return "setup1"
    if ram_gb >= 16 or vram_gb >= 8:
        return "basic"
    return "minimal"


def categorize_model(model_name: str, ram_gb: int, vram_gb: int) -> ModelFit:
    """Same case logic as categorize_model() in check-ollama-models.sh."""

    setup = setup_tier(ram_gb, vram_gb)

    small = frozenset({"llama3.2:1b", "llama3.2:3b", "gemma2:2b", "phi3:mini"})
    if model_name in small:
        return "recommended" if ram_gb >= 8 else "usable"

    seven_b = frozenset(
        {
            "mistral:7b",
            "neural-chat:7b",
            "qwen2.5:7b",
            "gemma:7b",
            "yi:6b",
            "openchat:7b",
            "zephyr:7b",
            "nous-hermes:7b",
            "mythomax:7b",
            "dolphin-mistral:7b",
            "orca-mini:7b",
        }
    )
    if model_name in seven_b:
        if setup in {"setup5", "setup4", "setup3", "setup2"}:
            return "recommended"
        if vram_gb >= 8 and ram_gb >= 16:
            return "recommended"
        if vram_gb >= 8 or ram_gb >= 16:
            return "usable"
        if ram_gb >= 8:
            return "marginal"
        return "no"

    if model_name in {"llama3:8b", "llama3.1:8b"}:
        if setup in {"setup5", "setup4", "setup3", "setup2"}:
            return "recommended"
        if setup == "setup1":
            return "usable"
        if ram_gb >= 16:
            return "marginal"
        return "no"

    if model_name in {"gemma2:9b", "yi:9b"}:
        if setup in {"setup5", "setup4", "setup3", "setup2"}:
            return "recommended"
        if setup == "setup1":
            return "usable"
        if ram_gb >= 32:
            return "marginal"
        return "no"

    if model_name in {"mistral-nemo:12b", "falcon:11b"}:
        if setup in {"setup5", "setup4", "setup3", "setup2"}:
            return "recommended"
        if ram_gb >= 32 and vram_gb >= 8:
            return "usable"
        return "marginal"

    if model_name in {"vicuna:13b", "openhermes:13b"}:
        if setup in {"setup5", "setup4", "setup3"}:
            return "recommended"
        if ram_gb >= 32:
            return "usable"
        return "marginal"

    if model_name == "gemma2:27b":
        if setup == "setup5":
            return "recommended"
        if ram_gb >= 32 and vram_gb >= 10:
            return "marginal"
        return "no"

    if model_name == "mixtral:8x7b":
        if setup in {"setup5", "setup4"}:
            return "marginal"
        return "no"

    if model_name == "llama3.1:70b":
        return "marginal" if vram_gb >= 48 else "no"

    code7 = frozenset(
        {
            "codellama:7b",
            "deepseek-coder:6.7b",
            "qwen2.5-coder:7b",
            "starcoder2:7b",
            "wizardcoder:7b",
        }
    )
    if model_name in code7:
        if setup in {"setup5", "setup4", "setup3", "setup2"}:
            return "recommended"
        if vram_gb >= 8 and ram_gb >= 16:
            return "recommended"
        if vram_gb >= 8 or ram_gb >= 16:
            return "usable"
        return "marginal"

    if model_name in {"codellama:13b", "wizardcoder:13b"}:
        if setup in {"setup5", "setup4", "setup3", "setup2"}:
            return "recommended"
        if ram_gb >= 32:
            return "usable"
        return "marginal"

    if model_name == "phi3:medium":
        if setup in {"setup5", "setup4", "setup3", "setup2"}:
            return "usable"
        if setup == "setup1":
            return "usable"
        if ram_gb >= 32:
            return "marginal"
        return "no"

    # Default * in shell
    if setup == "setup5":
        return "recommended"
    if setup in {"setup4", "setup3"}:
        return "usable"
    if vram_gb >= 8 and ram_gb >= 16:
        return "usable"
    return "marginal"


# Curated list (order preserved) from check-ollama-models.sh MODEL_LIST — unique model ids.
_CURATED: list[tuple[str, str]] = [
    ("mistral:7b", "DEFAULT - Used by this project"),
    ("llama3:8b", "Alternative model"),
    ("llama3.2:1b", "Small, fast"),
    ("llama3.2:3b", "Balanced"),
    ("llama3.1:8b", "Latest Llama 3.1"),
    ("llama3.1:70b", "Large (48GB+ VRAM)"),
    ("mistral-nemo:12b", "Enhanced Mistral"),
    ("mixtral:8x7b", "Mixture of experts"),
    ("gemma2:2b", "Small, efficient"),
    ("gemma2:9b", "Balanced"),
    ("gemma2:27b", "Large"),
    ("gemma:7b", "Google Gemma"),
    ("phi3:mini", "Small, fast"),
    ("phi3:medium", "Balanced"),
    ("codellama:7b", "Code generation"),
    ("codellama:13b", "Larger code model"),
    ("deepseek-coder:6.7b", "Advanced coding"),
    ("qwen2.5:7b", "Alibaba"),
    ("qwen2.5-coder:7b", "Alibaba coding"),
    ("neural-chat:7b", "Conversational"),
    ("yi:6b", "Yi 6B"),
    ("yi:9b", "Yi 9B"),
    ("openchat:7b", "OpenChat"),
    ("zephyr:7b", "Zephyr"),
    ("nous-hermes:7b", "Nous Hermes"),
    ("mythomax:7b", "MythoMax"),
    ("dolphin-mistral:7b", "Dolphin Mistral"),
    ("orca-mini:7b", "Orca Mini"),
    ("vicuna:13b", "Vicuna 13B"),
    ("falcon:11b", "Falcon 11B"),
    ("openhermes:13b", "OpenHermes"),
    ("starcoder2:7b", "StarCoder2"),
    ("wizardcoder:7b", "WizardCoder 7B"),
    ("wizardcoder:13b", "WizardCoder 13B"),
]


def curated_model_ids() -> list[str]:
    """Return curated Ollama model ids (stable order from check-ollama-models.sh)."""

    return [model_id for model_id, _desc in _CURATED]


def curated_models() -> list[tuple[str, str]]:
    """Return curated Ollama model ids and descriptions (stable order)."""

    return list(_CURATED)


def recommend_models(*, backend_hint: str | None = None) -> OllamaRecommendation:
    """Hardware-aware recommendations (no docker exec).

    For interactive container pulls use check-ollama-models.sh.
    """

    d = detect()
    ram = system_ram_gb()
    vram, notes = apply_backend_hint_vram(effective_vram_gb(d, ram), ram, backend_hint)

    tier = setup_tier(ram, vram)
    rec: list[str] = []
    usable: list[str] = []
    marginal: list[str] = []
    nope: list[str] = []

    for model_id, _desc in _CURATED:
        cat = categorize_model(model_id, ram, vram)
        if cat == "recommended":
            rec.append(model_id)
        elif cat == "usable":
            usable.append(model_id)
        elif cat == "marginal":
            marginal.append(model_id)
        else:
            nope.append(model_id)

    default_model = "mistral:7b"
    if categorize_model(default_model, ram, vram) in ("recommended", "usable"):
        pass
    elif rec:
        default_model = rec[0]
    elif usable:
        default_model = usable[0]
    else:
        if "llama3.2:1b" in marginal:
            default_model = "llama3.2:1b"
        elif marginal:
            default_model = marginal[0]
        else:
            default_model = "mistral:7b"

    notes.append(
        "For interactive pull flow inside Docker see diri-cyrex/scripts/llm/check-ollama-models.sh."
    )

    return OllamaRecommendation(
        default_model=default_model,
        recommended_models=rec,
        usable_models=usable,
        marginal_models=marginal,
        unsuitable_models=nope,
        notes=notes,
        setup_tier=tier,
        system_ram_gb=ram,
        effective_vram_gb=vram,
    )
