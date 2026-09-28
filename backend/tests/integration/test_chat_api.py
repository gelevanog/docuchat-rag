from __future__ import annotations

import uuid
from typing import Any

import httpx

from app.llm.prompts import NO_ANSWER
from tests.helpers import parse_sse


async def ask(client: httpx.AsyncClient, message: str, **extra: Any) -> dict[str, Any]:
    response = await client.post("/api/chat", json={"message": message, **extra})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert names[1] == "sources"
    assert names[-1] in {"done", "error"}
    by_name = {name: data for name, data in events if name != "token"}
    by_name["tokens"] = [data["text"] for name, data in events if name == "token"]
    return by_name


async def test_chat_streams_grounded_answer_with_citations(
    client: httpx.AsyncClient, handbook: dict[str, Any], security: dict[str, Any]
) -> None:
    result = await ask(client, "How many vacation days do full-time employees get?")

    sources = result["sources"]["sources"]
    assert sources, "retrieval should return chunks"
    assert sources[0]["document_id"] == handbook["id"]
    assert [s["id"] for s in sources] == list(range(1, len(sources) + 1))

    done = result["done"]
    assert "25 days of paid vacation" in done["answer"]
    assert "".join(result["tokens"]).strip() == done["answer"]
    assert done["citations"], "the answer should cite at least one source"
    cited = done["citations"][0]
    assert cited["document_title"] == "Acme Handbook"
    assert cited["heading"] == "Vacation"
    assert f"[{cited['id']}]" in done["answer"]


async def test_conversation_is_persisted_and_follow_ups_are_rewritten(
    client: httpx.AsyncClient, handbook: dict[str, Any]
) -> None:
    first = await ask(client, "What is the parental leave policy?")
    conversation_id = first["meta"]["conversation_id"]

    follow_up = await ask(client, "And for adoptive parents?", conversation_id=conversation_id)
    assert follow_up["meta"]["conversation_id"] == conversation_id
    assert "parental leave" in follow_up["meta"]["rewritten_query"]
    assert "10 weeks" in follow_up["done"]["answer"]

    conversations = (await client.get("/api/conversations")).json()
    assert [c["id"] for c in conversations] == [conversation_id]
    assert conversations[0]["title"] == "What is the parental leave policy?"

    detail = (await client.get(f"/api/conversations/{conversation_id}")).json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][2]["rewritten_query"] is not None
    assert detail["messages"][3]["citations"]

    assert (await client.delete(f"/api/conversations/{conversation_id}")).status_code == 204
    assert (await client.get(f"/api/conversations/{conversation_id}")).status_code == 404


async def test_document_filter_restricts_retrieval(
    client: httpx.AsyncClient, handbook: dict[str, Any], security: dict[str, Any]
) -> None:
    result = await ask(
        client, "How long must passwords be?", document_ids=[handbook["id"]], top_k=3
    )
    sources = result["sources"]["sources"]
    assert sources
    assert {s["document_id"] for s in sources} == {handbook["id"]}
    assert len(sources) <= 3


async def test_without_documents_the_assistant_says_it_does_not_know(
    client: httpx.AsyncClient,
) -> None:
    result = await ask(client, "What is the vacation policy?")
    assert result["sources"]["sources"] == []
    assert result["done"]["answer"] == NO_ANSWER
    assert result["done"]["citations"] == []


async def test_unknown_conversation_returns_404(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/chat", json={"message": "Hi", "conversation_id": str(uuid.uuid4())}
    )
    assert response.status_code == 404


async def test_request_validation(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/chat", json={"message": ""})).status_code == 422
    assert (await client.post("/api/chat", json={"message": "q", "top_k": 99})).status_code == 422
