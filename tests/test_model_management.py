import asyncio
from types import SimpleNamespace

from deepiri_ollama.client import OllamaAPIError, OllamaReadiness, PullResult
from deepiri_ollama import model_management as management


def _assessment(model="mistral:7b", warning=None):
    return management.ModelAssessment(
        model=model,
        fit=SimpleNamespace(model=model, fit="recommended"),
        workload=SimpleNamespace(model=model),
        capacity=SimpleNamespace(model=model),
        warning=warning,
    )


def test_recommendations_use_gpu_utils_results(monkeypatch):
    recommendation = SimpleNamespace(
        default_model="mistral:7b",
        system_ram_gb=32,
        effective_vram_gb=16,
        notes=["gpu-utils result"],
    )
    monkeypatch.setattr(management, "recommend_models", lambda **kwargs: recommendation)
    monkeypatch.setattr(
        management,
        "curated_models",
        lambda: [("mistral:7b", "General purpose"), ("llama3.2:1b", "Small")],
    )
    monkeypatch.setattr(
        management,
        "curated_model_ids",
        lambda: ["mistral:7b", "llama3.2:1b"],
    )
    monkeypatch.setattr(
        management,
        "categorize_model",
        lambda model, ram, vram: "recommended" if model == "mistral:7b" else "usable",
    )
    monkeypatch.setattr(management, "setup_tier", lambda ram, vram: "setup5")

    class Client:
        async def readiness(self):
            return OllamaReadiness(
                ready=True,
                base_url="http://localhost:11434",
                models=["mistral:7b"],
            )

    report = asyncio.run(management.recommendation_report(Client()))
    assert report.default_model == "mistral:7b"
    assert report.setup_tier == "setup5"
    assert report.rows[0].installed is True
    assert report.rows[1].fit == "usable"


def test_assess_model_uses_fit_workload_and_capacity_primitives(monkeypatch):
    calls = {}
    fit = SimpleNamespace(
        model="mistral:7b",
        fit="marginal",
        suitable=False,
        reason="tight fit",
    )
    workload = SimpleNamespace(model="mistral:7b")
    capacity = SimpleNamespace(model="mistral:7b")

    monkeypatch.setattr(
        management,
        "model_fit_check",
        lambda model, backend_hint=None: calls.setdefault(
            "fit", (model, backend_hint)
        )
        and fit,
    )
    monkeypatch.setattr(
        management,
        "estimate_workload",
        lambda model, context_tokens=4096: calls.setdefault(
            "workload", (model, context_tokens)
        )
        and workload,
    )
    monkeypatch.setattr(
        management,
        "model_capacity",
        lambda model, reserved_gb=1.0, context_tokens=4096: calls.setdefault(
            "capacity", (model, reserved_gb, context_tokens)
        )
        and capacity,
    )

    result = management.assess_model(
        "mistral:7b",
        backend_hint="cuda",
        context_tokens=8192,
        reserved_gb=2.0,
    )
    assert result.fit is fit
    assert result.warning == (
        "mistral:7b is marginal; inference may be slow or unstable."
    )
    assert calls == {
        "fit": ("mistral:7b", "cuda"),
        "workload": ("mistral:7b", 8192),
        "capacity": ("mistral:7b", 2.0, 8192),
    }


def test_install_model_pulls_and_verifies(monkeypatch):
    monkeypatch.setattr(management, "assess_model", lambda *args, **kwargs: _assessment())
    answers = iter([False, True])

    class Client:
        async def has_model(self, model):
            return next(answers)

        async def pull_model(self, model):
            return PullResult(model=model, success=True, statuses=[{"status": "success"}])

    result = asyncio.run(management.install_model(Client(), "mistral:7b"))
    assert result.success is True
    assert result.pulled is True
    assert result.verified is True


def test_install_model_reports_pull_failure(monkeypatch):
    monkeypatch.setattr(management, "assess_model", lambda *args, **kwargs: _assessment())

    class Client:
        async def has_model(self, model):
            return False

        async def pull_model(self, model):
            return PullResult(
                model=model,
                success=False,
                error="manifest not found",
            )

    result = asyncio.run(management.install_model(Client(), "mistral:7b"))
    assert result.success is False
    assert result.verified is False
    assert result.error == "manifest not found"


def test_install_model_reports_api_failure(monkeypatch):
    monkeypatch.setattr(management, "assess_model", lambda *args, **kwargs: _assessment())

    class Client:
        async def has_model(self, model):
            raise OllamaAPIError("connection refused")

    result = asyncio.run(management.install_model(Client(), "mistral:7b"))
    assert result.success is False
    assert result.error == "connection refused"


def test_remove_model_deletes_and_verifies():
    answers = iter([True, False])
    deleted = []

    class Client:
        async def has_model(self, model):
            return next(answers)

        async def delete_model(self, model):
            deleted.append(model)
            return True

    result = asyncio.run(management.remove_model(Client(), "mistral:7b"))
    assert result.success is True
    assert result.existed is True
    assert result.verified_absent is True
    assert deleted == ["mistral:7b"]
