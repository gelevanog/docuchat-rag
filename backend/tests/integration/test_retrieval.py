from __future__ import annotations

import uuid
from typing import Any

import httpx
from fastapi import FastAPI

from app.container import Container


def container_of(app: FastAPI) -> Container:
    container: Container = app.state.container
    return container


async def test_hybrid_search_ranks_the_answering_chunk_first(
    app: FastAPI, handbook: dict[str, Any], security: dict[str, Any]
) -> None:
    container = container_of(app)
    async with container.sessionmaker() as session:
        results = await container.retriever.search(session, "minimum password length", top_k=3)
    assert results[0].document_id == uuid.UUID(security["id"])
    assert "14 characters" in results[0].content
    assert results[0].heading == "Passwords"
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


async def test_keyword_search_matches_word_variants(
    app: FastAPI, handbook: dict[str, Any], security: dict[str, Any]
) -> None:
    container = container_of(app)
    async with container.sessionmaker() as session:
        # "adopting" and "adoptive" share the stem "adopt" in PostgreSQL's English config.
        ids = await container.retriever.keyword_search(session, "adopting a child")
        assert ids
        results = await container.retriever.search(session, "adopting a child", top_k=1)
    assert "Adoptive parents" in results[0].content
    assert results[0].keyword_rank == 1


async def test_search_ignores_documents_that_are_not_ready(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    container = container_of(app)
    async with container.sessionmaker() as session:
        assert await container.retriever.search(session, "anything at all", top_k=5) == []
