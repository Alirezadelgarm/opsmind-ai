import asyncio

import httpx

from opsmind.api.app import create_app
from opsmind.config import load_settings


def test_health_endpoint() -> None:
    settings = load_settings()
    app = create_app(settings)

    async def request_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "1.0.2"
    assert response.headers["x-request-id"]
