"""Wires settings, database and providers into the application's services."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.session import create_engine, create_sessionmaker
from app.ingestion.chunking import RecursiveTokenSplitter, TiktokenTokenizer
from app.ingestion.pipeline import IngestionPipeline
from app.llm.base import ChatModel, EmbeddingModel
from app.llm.factory import build_chat_model, build_embedding_model
from app.retrieval.rerank import Reranker, build_reranker
from app.retrieval.search import HybridRetriever
from app.services.chat import ChatService
from app.services.documents import DocumentService
from app.services.query_rewriter import build_query_rewriter


@dataclass(slots=True)
class Container:
    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    chat_model: ChatModel
    embedder: EmbeddingModel
    reranker: Reranker | None
    retriever: HybridRetriever
    pipeline: IngestionPipeline
    documents: DocumentService
    chat: ChatService

    @classmethod
    def build(cls, settings: Settings) -> Container:
        engine = create_engine(settings)
        sessionmaker = create_sessionmaker(engine)
        chat_model = build_chat_model(settings)
        embedder = build_embedding_model(settings)
        splitter = RecursiveTokenSplitter(
            TiktokenTokenizer(),
            chunk_size=settings.chunk_size_tokens,
            chunk_overlap=settings.chunk_overlap_tokens,
        )
        reranker = build_reranker(settings.reranker, settings)
        retriever = HybridRetriever(
            embedder,
            candidates=settings.retrieval_candidates,
            rrf_k=settings.rrf_k,
            reranker=reranker,
            rerank_candidates=settings.rerank_candidates,
        )
        return cls(
            settings=settings,
            engine=engine,
            sessionmaker=sessionmaker,
            chat_model=chat_model,
            embedder=embedder,
            reranker=reranker,
            retriever=retriever,
            pipeline=IngestionPipeline(
                sessionmaker, embedder, splitter, batch_size=settings.embedding_batch_size
            ),
            documents=DocumentService(settings.upload_dir, settings.max_upload_bytes),
            chat=ChatService(
                chat_model,
                build_query_rewriter(settings.llm_provider, chat_model),
                retriever,
                default_top_k=settings.retrieval_top_k,
                history_turns=settings.history_turns,
            ),
        )

    async def aclose(self) -> None:
        await self.engine.dispose()
