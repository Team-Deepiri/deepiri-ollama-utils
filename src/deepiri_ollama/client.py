"""Reusable asynchronous client for the Ollama HTTP API."""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaAPIError(RuntimeError):
    """Raised when Ollama cannot complete an API operation."""


@dataclass(frozen=True)
class OllamaModelInfo:
    """Metadata returned by Ollama's ``/api/tags`` endpoint."""

    name: str
    size: int = 0
    digest: str = ""
    modified_at: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PullResult:
    """Outcome and progress events from an Ollama model pull."""

    model: str
    success: bool
    statuses: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class OllamaReadiness:
    """Reachability and installed-model snapshot for an Ollama server."""

    ready: bool
    base_url: str
    models: list[str] = field(default_factory=list)
    error: str | None = None


ProgressCallback = Callable[[dict[str, Any]], None | Awaitable[None]]


def resolve_base_url(base_url: str | None = None) -> str:
    """Resolve an explicit URL, then ``OLLAMA_BASE_URL``, then localhost."""

    return (base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


class OllamaClient:
    """Application-neutral client for model management in Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = resolve_base_url(base_url)
        self.timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OllamaClient:
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                transport=self._transport,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        client = await self._get_client()
        try:
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaAPIError(f"Ollama {method} {path} failed: {exc}") from exc

        if not isinstance(data, dict):
            raise OllamaAPIError(f"Ollama {method} {path} returned a non-object response")
        return data

    async def list_models(self) -> list[OllamaModelInfo]:
        """Return models installed on the configured Ollama server."""

        data = await self._request_json("GET", "/api/tags")
        models: list[OllamaModelInfo] = []
        for raw in data.get("models", []):
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                continue
            size = raw.get("size", 0)
            digest = raw.get("digest", "")
            modified_at = raw.get("modified_at", "")
            details = raw.get("details")
            models.append(
                OllamaModelInfo(
                    name=raw["name"],
                    size=size if isinstance(size, int) else 0,
                    digest=digest if isinstance(digest, str) else "",
                    modified_at=modified_at if isinstance(modified_at, str) else "",
                    details=details if isinstance(details, dict) else {},
                )
            )
        return models

    async def has_model(self, model_name: str) -> bool:
        """Return whether ``model_name`` is installed."""

        return model_name.strip() in {model.name for model in await self.list_models()}

    async def show_model(self, model_name: str) -> dict[str, Any]:
        """Return detailed Ollama metadata for one model."""

        return await self._request_json("POST", "/api/show", json={"name": model_name.strip()})

    async def pull_model(
        self,
        model_name: str,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> PullResult:
        """Pull a model and collect Ollama's newline-delimited progress events."""

        model = model_name.strip()
        client = await self._get_client()
        statuses: list[dict[str, Any]] = []

        try:
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": model, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaAPIError(
                            "Ollama pull returned invalid progress JSON"
                        ) from exc
                    if not isinstance(event, dict):
                        continue
                    statuses.append(event)
                    if progress_callback is not None:
                        callback_result = progress_callback(event)
                        if inspect.isawaitable(callback_result):
                            await callback_result
                    if event.get("error"):
                        return PullResult(
                            model=model,
                            success=False,
                            statuses=statuses,
                            error=str(event["error"]),
                        )
        except httpx.HTTPError as exc:
            raise OllamaAPIError(f"Ollama pull failed for {model!r}: {exc}") from exc

        return PullResult(model=model, success=True, statuses=statuses)

    async def delete_model(self, model_name: str) -> bool:
        """Delete a model through Ollama's model-management API."""

        client = await self._get_client()
        model = model_name.strip()
        try:
            response = await client.request("DELETE", "/api/delete", json={"name": model})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaAPIError(f"Ollama delete failed for {model!r}: {exc}") from exc
        return True

    async def readiness(self) -> OllamaReadiness:
        """Return a non-raising readiness result for the configured server."""

        try:
            models = await self.list_models()
        except OllamaAPIError as exc:
            return OllamaReadiness(ready=False, base_url=self.base_url, error=str(exc))
        return OllamaReadiness(
            ready=True,
            base_url=self.base_url,
            models=[model.name for model in models],
        )
