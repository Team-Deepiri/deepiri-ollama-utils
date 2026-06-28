import asyncio
import json
import subprocess

import httpx

from deepiri_ollama import cli
from deepiri_ollama.runtime import check, list_models, verify_models


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _patch_async_client(monkeypatch, response=None, error=None):
    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url):
            if error is not None:
                raise error
            return response

    monkeypatch.setattr("deepiri_ollama.runtime.httpx.AsyncClient", DummyAsyncClient)


def _connect_error():
    request = httpx.Request("GET", "http://localhost:11434/api/tags")
    return httpx.ConnectError("Connection refused", request=request)


def test_check_returns_expected_shape_when_ollama_is_down(monkeypatch):
    _patch_async_client(monkeypatch, error=_connect_error())

    result = asyncio.run(check("http://localhost:11434"))

    assert isinstance(result, dict)
    assert result["ok"] is False
    assert result["running"] is False
    assert result["base_url"] == "http://localhost:11434"
    assert result["models"] == []
    assert result["message"] == "Ollama not running"


def test_list_models_returns_list(monkeypatch):
    _patch_async_client(
        monkeypatch,
        response=DummyResponse(
            payload={"models": [{"name": "llama2"}, {"name": "mistral"}]}
        ),
    )

    result = asyncio.run(list_models("http://localhost:11434"))

    assert result == ["llama2", "mistral"]


def test_verify_models_reports_missing_models(monkeypatch):
    _patch_async_client(
        monkeypatch,
        response=DummyResponse(payload={"models": [{"name": "llama2"}]}),
    )

    result = asyncio.run(
        verify_models(["llama2", "mistral"], "http://localhost:11434")
    )

    assert result["ok"] is False
    assert result["running"] is True
    assert result["base_url"] == "http://localhost:11434"
    assert result["requested"] == ["llama2", "mistral"]
    assert result["available"] == ["llama2"]
    assert result["missing"] == ["mistral"]
    assert result["all_present"] is False
    assert result["message"] == "Missing 1 model(s)"


def test_cli_verify_models_prints_json(monkeypatch, capsys):
    _patch_async_client(
        monkeypatch,
        response=DummyResponse(payload={"models": [{"name": "llama2"}]}),
    )

    cli.main(["verify-models", "--model", "llama2", "--model", "mistral"])

    output = json.loads(capsys.readouterr().out)
    assert output["requested"] == ["llama2", "mistral"]
    assert output["available"] == ["llama2"]
    assert output["missing"] == ["mistral"]
    assert output["all_present"] is False


def test_cli_verify_models_handles_connection_failure(monkeypatch, capsys):
    _patch_async_client(monkeypatch, error=_connect_error())

    cli.main(["verify-models", "--model", "llama2"])

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["running"] is False
    assert output["requested"] == ["llama2"]
    assert output["available"] == []
    assert output["missing"] == ["llama2"]
    assert output["all_present"] is False
    assert output["message"] == "Ollama not running"


def test_cli_verify_models_file_prints_json(monkeypatch, capsys, tmp_path):
    _patch_async_client(
        monkeypatch,
        response=DummyResponse(payload={"models": [{"name": "llama2"}]}),
    )
    models_file = tmp_path / "required-models.txt"
    models_file.write_text("# required\nllama2\n\nmistral\n", encoding="utf-8")

    cli.main(["verify-models-file", "--file", str(models_file)])

    output = json.loads(capsys.readouterr().out)
    assert output["requested"] == ["llama2", "mistral"]
    assert output["available"] == ["llama2"]
    assert output["missing"] == ["mistral"]
    assert output["all_present"] is False


def test_cli_verify_models_file_handles_missing_file(capsys):
    cli.main(["verify-models-file", "--file", "does-not-exist.txt"])

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["running"] is False
    assert output["requested"] == []
    assert output["available"] == []
    assert output["missing"] == []
    assert output["all_present"] is False
    assert "does-not-exist.txt" in output["message"]


def test_cli_verify_models_requires_model(capsys):
    cli.main(["verify-models"])

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["requested"] == []
    assert output["missing"] == []
    assert output["all_present"] is False
    assert output["message"] == "Missing --model"


def test_cli_delegates_gpu_commands_with_arguments_unchanged(monkeypatch):
    calls = []

    def fake_run(args, check):
        calls.append((args, check))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("deepiri_ollama.cli.subprocess.run", fake_run)
    forwarded_args = ["--format", "json", "--workload=coding", "--", "literal"]

    for command in cli.DELEGATED_COMMANDS:
        assert cli.main([command, *forwarded_args]) == 0

    assert calls == [
        (["deepiri-gpu", command, *forwarded_args], False)
        for command in cli.DELEGATED_COMMANDS
    ]


def test_cli_returns_delegated_nonzero_exit_code(monkeypatch):
    def fake_run(args, check):
        return subprocess.CompletedProcess(args, 23)

    monkeypatch.setattr("deepiri_ollama.cli.subprocess.run", fake_run)

    assert cli.main(["model-fit", "llama3:8b"]) == 23


def test_cli_reports_missing_deepiri_gpu(monkeypatch, capsys):
    def fake_run(args, check):
        raise FileNotFoundError

    monkeypatch.setattr("deepiri_ollama.cli.subprocess.run", fake_run)

    assert cli.main(["capacity"]) == 127
    assert "deepiri-gpu was not found" in capsys.readouterr().err
