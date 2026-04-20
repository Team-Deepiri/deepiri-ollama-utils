import httpx

DEFAULT_BASE_URL = "http://localhost:11434"


def _url(base_url, path):
    return f"{base_url.rstrip('/')}{path}"


def _normalize_model_names(model_names):
    """Normalize model names by stripping whitespace, dropping blanks, and removing duplicates."""
    if not model_names:
        return []

    normalized = []
    for model_name in model_names:
        if not isinstance(model_name, str):
            continue

        name = model_name.strip()
        if name and name not in normalized:
            normalized.append(name)

    return normalized


async def is_ollama_running(base_url=DEFAULT_BASE_URL):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(_url(base_url, "/api/tags"))
            return r.status_code == 200
    except Exception:
        return False


async def list_models(base_url=DEFAULT_BASE_URL):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(_url(base_url, "/api/tags"))
            data = r.json()
            return [m.get("name") for m in data.get("models", []) if isinstance(m.get("name"), str)]
    except Exception:
        return []


async def has_model(model_name, base_url=DEFAULT_BASE_URL):
    models = await list_models(base_url)
    return model_name in models


async def check(base_url=DEFAULT_BASE_URL):
    running = await is_ollama_running(base_url)

    if not running:
        return {
            "ok": False,
            "running": False,
            "base_url": base_url,
            "models": [],
            "message": "Ollama not running",
        }

    models = await list_models(base_url)

    return {
        "ok": True,
        "running": True,
        "base_url": base_url,
        "models": models,
        "message": f"Found {len(models)} models",
    }


async def verify_models(model_names, base_url=DEFAULT_BASE_URL):
    requested = _normalize_model_names(model_names)

    if not requested:
        return {
            "ok": False,
            "running": False,
            "base_url": base_url,
            "requested": [],
            "available": [],
            "missing": [],
            "all_present": False,
            "message": "Missing --model",
        }

    running = await is_ollama_running(base_url)

    if not running:
        return {
            "ok": False,
            "running": False,
            "base_url": base_url,
            "requested": requested,
            "available": [],
            "missing": requested,
            "all_present": False,
            "message": "Ollama not running",
        }

    available = await list_models(base_url)
    missing = [model_name for model_name in requested if model_name not in available]
    all_present = len(missing) == 0

    return {
        "ok": all_present,
        "running": True,
        "base_url": base_url,
        "requested": requested,
        "available": available,
        "missing": missing,
        "all_present": all_present,
        "message": (
            "All requested models are available"
            if all_present
            else f"Missing {len(missing)} model(s)"
        ),
    }