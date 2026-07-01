import asyncio
import json

import httpx
import pytest

from deepiri_ollama.client import (
    OllamaAPIError,
    OllamaClient,
    resolve_base_url,
)


def test_resolve_base_url_uses_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434/")
    assert resolve_base_url() == "http://ollama.internal:11434"


def test_list_show_and_readiness():
    async def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "mistral:7b",
                            "size": 123,
                            "digest": "abc",
                            "details": {"family": "mistral"},
                        }
                    ]
                },
            )
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"license": "apache"})
        return httpx.Response(404)

    async def run():
        async with OllamaClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            models = await client.list_models()
            assert [model.name for model in models] == ["mistral:7b"]
            assert models[0].details == {"family": "mistral"}
            assert await client.has_model("mistral:7b") is True
            assert await client.show_model("mistral:7b") == {"license": "apache"}
            readiness = await client.readiness()
            assert readiness.ready is True
            assert readiness.models == ["mistral:7b"]

    asyncio.run(run())


def test_pull_model_streams_progress():
    events = []

    async def handler(request):
        assert request.url.path == "/api/pull"
        assert json.loads(request.content) == {"name": "mistral:7b", "stream": True}
        return httpx.Response(
            200,
            text='{"status":"pulling manifest"}\n{"status":"success"}\n',
        )

    async def run():
        async with OllamaClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            result = await client.pull_model(
                "mistral:7b",
                progress_callback=events.append,
            )
            assert result.success is True
            assert [event["status"] for event in result.statuses] == [
                "pulling manifest",
                "success",
            ]

    asyncio.run(run())
    assert len(events) == 2


def test_pull_model_reports_stream_error():
    async def handler(request):
        return httpx.Response(200, text='{"error":"manifest not found"}\n')

    async def run():
        async with OllamaClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            result = await client.pull_model("missing:model")
            assert result.success is False
            assert result.error == "manifest not found"

    asyncio.run(run())


def test_pull_model_raises_for_http_failure():
    async def handler(request):
        return httpx.Response(500, json={"error": "server error"})

    async def run():
        async with OllamaClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(OllamaAPIError, match="pull failed"):
                await client.pull_model("mistral:7b")

    asyncio.run(run())


def test_delete_model_uses_ollama_api():
    seen = []

    async def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200)

    async def run():
        async with OllamaClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            assert await client.delete_model("mistral:7b") is True

    asyncio.run(run())
    assert seen == [("DELETE", "/api/delete", {"name": "mistral:7b"})]
