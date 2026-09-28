"""Hybrid retrieval: pgvector cosine similarity + PostgreSQL full-text search, fused with RRF."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, Select, Text, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Chunk, Document, DocumentStatus
from app.llm.base import EmbeddingModel
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk

logger = get_logger(__name__)

_TS_CONFIG = "english"


def _or_tsquery(query: str) -> ColumnElement[Any]:
    """`plainto_tsquery` ANDs every term, which is too strict for natural-language
    questions; rewrite its output into an OR query and let ts_rank_cd do the ordering."""
    plain = cast(func.plainto_tsquery(_TS_CONFIG, query), Text)
    return func.to_tsquery(_TS_CONFIG, func.replace(plain, "&", "|"))


class HybridRetriever:
    def __init__(self, embedder: EmbeddingModel, candidates: int = 30, rrf_k: int = 60) -> None:
        self._embedder = embedder
        self._candidates = candidates
        self._rrf_k = rrf_k

    @staticmethod
    def _candidate_ids(document_ids: Sequence[uuid.UUID] | None) -> Select[uuid.UUID]:
        """Chunk ids of ready documents, optionally restricted to `document_ids`."""
        stmt = (
            select(Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == DocumentStatus.READY)
        )
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        return stmt

    async def vector_search(
        self,
        session: AsyncSession,
        query_embedding: Sequence[float],
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[uuid.UUID]:
        # Make the HNSW scan return enough candidates, and keep scanning when a document
        # filter removes rows (pgvector >= 0.8 iterative scans; ignored on older versions).
        await session.execute(
            select(func.set_config("hnsw.ef_search", str(max(40, self._candidates)), True))
        )
        await session.execute(select(func.set_config("hnsw.iterative_scan", "relaxed_order", True)))
        distance = Chunk.embedding.cosine_distance(list(query_embedding))
        stmt = self._candidate_ids(document_ids).order_by(distance).limit(self._candidates)
        return list(await session.scalars(stmt))

    async def keyword_search(
        self,
        session: AsyncSession,
        query: str,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[uuid.UUID]:
        tsquery = _or_tsquery(query)
        rank = func.ts_rank_cd(Chunk.search_vector, tsquery)
        stmt = (
            self._candidate_ids(document_ids)
            .where(Chunk.search_vector.op("@@")(tsquery))
            .order_by(rank.desc(), Chunk.id)
            .limit(self._candidates)
        )
        return list(await session.scalars(stmt))

    async def search(
        self,
        session: AsyncSession,
        query: str,
        top_k: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        [query_embedding] = await self._embedder.embed([query])
        vector_ids = await self.vector_search(session, query_embedding, document_ids)
        keyword_ids = await self.keyword_search(session, query, document_ids)

        fused = reciprocal_rank_fusion([vector_ids, keyword_ids], k=self._rrf_k)[:top_k]
        if not fused:
            return []

        rows = await session.execute(
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.content,
                Chunk.heading,
                Chunk.page,
                Document.title,
                Document.filename,
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_([chunk_id for chunk_id, _ in fused]))
        )
        by_id = {row.id: row for row in rows}
        vector_rank = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids, start=1)}
        keyword_rank = {chunk_id: rank for rank, chunk_id in enumerate(keyword_ids, start=1)}

        results = []
        for chunk_id, score in fused:
            row = by_id[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    document_title=row.title,
                    filename=row.filename,
                    content=row.content,
                    heading=row.heading,
                    page=row.page,
                    score=score,
                    vector_rank=vector_rank.get(chunk_id),
                    keyword_rank=keyword_rank.get(chunk_id),
                )
            )
        logger.debug(
            "hybrid_search",
            query=query,
            vector_hits=len(vector_ids),
            keyword_hits=len(keyword_ids),
            returned=len(results),
        )
        return results
