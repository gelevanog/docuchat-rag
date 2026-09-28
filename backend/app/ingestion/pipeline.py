"""Ingestion pipeline: parse -> clean -> chunk -> embed (batched) -> store in pgvector."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.db.models import Chunk, Document, DocumentStatus
from app.ingestion.chunking import RecursiveTokenSplitter, TextChunk, chunk_blocks
from app.ingestion.parsers import parse_document
from app.llm.base import EmbeddingModel

logger = get_logger(__name__)

_MAX_ERROR_CHARS = 1000


def embedding_input(document_title: str, chunk: TextChunk) -> str:
    """Prefix each chunk with its document title and section so embeddings carry context."""
    header = document_title if not chunk.heading else f"{document_title} - {chunk.heading}"
    return f"{header}\n\n{chunk.text}"


class IngestionPipeline:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        embedder: EmbeddingModel,
        splitter: RecursiveTokenSplitter,
        batch_size: int,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._embedder = embedder
        self._splitter = splitter
        self._batch_size = batch_size

    async def embed_chunks(self, title: str, chunks: Sequence[TextChunk]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors.extend(await self._embedder.embed([embedding_input(title, c) for c in batch]))
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding provider returned an unexpected number of vectors")
        return vectors

    async def ingest(self, document_id: uuid.UUID) -> None:
        """Process one document. Never raises: failures are recorded on the document row."""
        log = logger.bind(document_id=str(document_id))
        started = time.perf_counter()
        async with self._sessionmaker() as session:
            document = await session.get(Document, document_id)
            if document is None:
                log.warning("ingest_document_missing")
                return
            document.status = DocumentStatus.PROCESSING
            document.error = None
            await session.commit()

            try:
                data = await asyncio.to_thread(Path(document.storage_path).read_bytes)
                parsed = await asyncio.to_thread(parse_document, data, document.filename)
                chunks = await asyncio.to_thread(chunk_blocks, parsed.blocks, self._splitter)
                vectors = await self.embed_chunks(parsed.title, chunks)

                # Idempotent: a retry replaces whatever a previous attempt stored.
                await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
                session.add_all(
                    Chunk(
                        document_id=document_id,
                        chunk_index=chunk.index,
                        content=chunk.text,
                        heading=chunk.heading,
                        page=chunk.page,
                        token_count=chunk.token_count,
                        embedding=vector,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                )
                document.title = parsed.title
                document.page_count = parsed.page_count
                document.chunk_count = len(chunks)
                document.status = DocumentStatus.READY
                await session.commit()
            except Exception as exc:  # the background task must record any failure
                await session.rollback()
                log.exception("ingest_failed")
                await self._mark_failed(session, document_id, exc)
                return

            log.info(
                "ingest_completed",
                chunks=len(chunks),
                pages=parsed.page_count,
                seconds=round(time.perf_counter() - started, 3),
            )

    @staticmethod
    async def _mark_failed(session: AsyncSession, document_id: uuid.UUID, exc: Exception) -> None:
        document = await session.get(Document, document_id)
        if document is None:
            return
        document.status = DocumentStatus.FAILED
        document.error = (str(exc) or type(exc).__name__)[:_MAX_ERROR_CHARS]
        await session.commit()
