from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from app.container import Container
from app.retrieval.rerank import LexicalReranker
from app.retrieval.search import HybridRetriever
from app.retrieval.types import RetrievedChunk


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


class RecordingReranker:
    """Lexical re-ranking that remembers the size of every candidate pool it was given."""

    def __init__(self) -> None:
        self.pools: list[int] = []
        self._inner = LexicalReranker()

    @property
    def name(self) -> str:
        return "recording"

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        self.pools.append(len(candidates))
        return await self._inner.rerank(query, candidates, top_k)


async def test_reranker_sees_a_larger_candidate_pool_and_returns_top_k(
    app: FastAPI, handbook: dict[str, Any], security: dict[str, Any]
) -> None:
    container = container_of(app)
    recorder = RecordingReranker()
    retriever = HybridRetriever(container.embedder, reranker=recorder, rerank_candidates=3)
    async with container.sessionmaker() as session:
        results = await retriever.search(session, "adoptive parents", top_k=1)
        # A request for more chunks than RERANK_CANDIDATES widens the pool to top_k.
        wide = await retriever.search(session, "adoptive parents", top_k=4)

    assert recorder.pools == [3, 4]
    assert len(results) == 1
    assert "Adoptive parents" in results[0].content
    assert results[0].rerank_score == pytest.approx(1.0)
    assert results[0].score > 0  # the fusion score is kept alongside
    assert [r.rerank_score for r in wide] == sorted((r.rerank_score for r in wide), reverse=True)


async def test_without_a_reranker_scores_stay_unset(app: FastAPI, handbook: dict[str, Any]) -> None:
    container = container_of(app)
    async with container.sessionmaker() as session:
        results = await container.retriever.search(session, "vacation", top_k=2)
    assert results
    assert all(r.rerank_score is None for r in results)
