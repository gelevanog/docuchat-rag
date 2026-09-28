from __future__ import annotations

import httpx


async def test_health_reports_database_and_providers(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["llm_provider"] == "fake:extractive"
    assert body["embedding_provider"].startswith("fake:hashing")


async def test_openapi_schema_lists_all_endpoints(client: httpx.AsyncClient) -> None:
    paths = (await client.get("/openapi.json")).json()["paths"]
    assert {"/api/documents", "/api/chat", "/api/conversations", "/health"} <= set(paths)
