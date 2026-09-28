from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from app.core.config import Settings
from tests.helpers import make_pdf
from tests.integration.conftest import HANDBOOK, upload


async def test_upload_ingests_document_in_background(client: httpx.AsyncClient) -> None:
    body = await upload(client, "handbook.md", HANDBOOK)
    assert body["duplicate"] is False
    assert body["document"]["status"] == "pending"

    # ASGITransport runs background tasks before returning, so ingestion has finished.
    document = (await client.get(f"/api/documents/{body['document']['id']}")).json()
    assert document["status"] == "ready"
    assert document["title"] == "Acme Handbook"
    assert document["chunk_count"] >= 2
    assert document["error"] is None


async def test_reupload_of_identical_content_is_a_noop(
    client: httpx.AsyncClient, handbook: dict[str, Any]
) -> None:
    again = await upload(client, "renamed-copy.md", HANDBOOK)
    assert again["duplicate"] is True
    assert again["document"]["id"] == handbook["id"]
    assert len((await client.get("/api/documents")).json()) == 1


async def test_pdf_chunks_keep_page_numbers(client: httpx.AsyncClient) -> None:
    pdf = make_pdf(
        ["Remote work is allowed from Portugal.", "Hotels are reimbursed up to 150 EUR."]
    )
    document = (await upload(client, "policy.pdf", pdf))["document"]
    document = (await client.get(f"/api/documents/{document['id']}")).json()
    assert document["status"] == "ready"
    assert document["page_count"] == 2


async def test_unparseable_file_is_marked_failed(client: httpx.AsyncClient) -> None:
    document = (await upload(client, "scan.pdf", make_pdf([""])))["document"]
    document = (await client.get(f"/api/documents/{document['id']}")).json()
    assert document["status"] == "failed"
    assert "no text layer" in document["error"]


async def test_rejects_unsupported_and_empty_files(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/documents", files={"file": ("photo.png", b"\x89PNG")})
    assert response.status_code == 415
    assert "Unsupported" in response.json()["detail"]
    response = await client.post("/api/documents", files={"file": ("empty.txt", b"")})
    assert response.status_code == 415


async def test_list_get_and_delete(
    client: httpx.AsyncClient, handbook: dict[str, Any], settings: Settings
) -> None:
    listing = (await client.get("/api/documents")).json()
    assert [d["id"] for d in listing] == [handbook["id"]]
    stored_file = Path(settings.upload_dir) / f"{handbook['id']}.md"
    assert stored_file.exists()

    assert (await client.delete(f"/api/documents/{handbook['id']}")).status_code == 204
    assert (await client.get(f"/api/documents/{handbook['id']}")).status_code == 404
    assert (await client.delete(f"/api/documents/{handbook['id']}")).status_code == 404
    assert (await client.get("/api/documents")).json() == []
    assert not stored_file.exists()
