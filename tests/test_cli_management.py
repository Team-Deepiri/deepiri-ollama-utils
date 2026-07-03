import json
from types import SimpleNamespace

from deepiri_ollama.model_fit import ModelFitResult

from deepiri_ollama import cli
from deepiri_ollama.model_management import (
    InstallResult,
    ModelAssessment,
    RecommendationReport,
    RecommendationRow,
    RemoveResult,
)


def test_cli_recommend_models_json(monkeypatch, capsys):
    report = RecommendationReport(
        default_model="mistral:7b",
        setup_tier="setup5",
        system_ram_gb=32,
        effective_vram_gb=16,
        ready=True,
        base_url="http://localhost:11434",
        installed_models=["mistral:7b"],
        rows=[
            RecommendationRow(
                model="mistral:7b",
                description="General purpose",
                fit="recommended",
                suitable=True,
                installed=True,
            )
        ],
    )

    async def fake_recommend(args):
        return report

    monkeypatch.setattr(cli, "_recommend", fake_recommend)
    assert cli.main(["recommend-models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["default_model"] == "mistral:7b"
    assert payload["rows"][0]["installed"] is True


def test_cli_install_model_displays_fit_warning(monkeypatch, capsys):
    assessment = ModelAssessment(
        model="mistral:7b",
        fit=SimpleNamespace(model="mistral:7b"),
        workload=SimpleNamespace(model="mistral:7b"),
        capacity=SimpleNamespace(model="mistral:7b"),
        warning="mistral:7b is marginal",
    )
    result = InstallResult(
        model="mistral:7b",
        success=True,
        already_installed=False,
        pulled=True,
        verified=True,
        assessment=assessment,
    )

    async def fake_install(args, model):
        return result

    monkeypatch.setattr(cli, "_install", fake_install)
    assert cli.main(["install-model", "--model", "mistral:7b"]) == 0
    output = capsys.readouterr().out
    assert "Warning: mistral:7b is marginal" in output
    assert "verified=True" in output


def test_cli_remove_model_json(monkeypatch, capsys):
    async def fake_remove(args, model):
        return RemoveResult(
            model=model,
            success=True,
            existed=True,
            verified_absent=True,
        )

    monkeypatch.setattr(cli, "_remove", fake_remove)
    assert cli.main(["remove-model", "--model", "mistral:7b", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["verified_absent"] is True


def test_cli_model_fit_calls_gpu_primitive(monkeypatch, capsys):
    seen = {}

    def fake_model_fit(model, backend_hint=None):
        seen["args"] = (model, backend_hint)
        return ModelFitResult(
            model=model,
            fit="usable",
            setup_tier="setup1",
            system_ram_gb=16,
            effective_vram_gb=8,
            default_model="mistral:7b",
            suitable=True,
            reason="usable",
            notes=[],
        )

    monkeypatch.setattr(
        "deepiri_ollama.model_fit.model_fit_check",
        fake_model_fit,
    )
    assert (
        cli.main(
            [
                "model-fit",
                "--model",
                "mistral:7b",
                "--backend-hint",
                "cuda",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["fit"] == "usable"
    assert seen["args"] == ("mistral:7b", "cuda")


def test_existing_cli_error_exit_semantics_remain_compatible(capsys):
    assert cli.main(["verify-models"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["message"] == "Missing --model"
