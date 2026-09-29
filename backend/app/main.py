"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import chat, conversations, documents, health
from app.container import Container
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        container = Container.build(settings)
        app.state.container = container
        logger.info(
            "startup",
            llm=container.chat_model.name,
            embeddings=container.embedder.name,
            reranker=container.reranker.name if container.reranker else None,
            environment=settings.environment,
        )
        try:
            yield
        finally:
            await container.aclose()

    app = FastAPI(
        title="DocuChat API",
        version=__version__,
        description=(
            "Retrieval-augmented chat over your documents: upload files, ask questions, "
            "get streamed answers with citations to the exact source chunks."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (health, documents, chat, conversations):
        app.include_router(module.router)
    return app
