import httpx

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    async def list_models(self):
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{self.base_url}/api/tags")
            return r.json()