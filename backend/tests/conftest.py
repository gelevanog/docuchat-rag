"""Shared fixtures.

Unit tests need nothing external. Integration tests need PostgreSQL with pgvector: point
TEST_DATABASE_URL at it (default: the `make test-db` container) or they are skipped (set
REQUIRE_TEST_DB=1 to fail instead, as CI does). The test database schema is dropped and
rebuilt with Alembic at the start of the session.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings
from app.main import create_app

BACKEND_DIR = Path(__file__).resolve().parents[1]
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://docuchat:docuchat@127.0.0.1:55433/docuchat_test"
)
TEST_EMBEDDING_DIM = 256


async def _reset_schema(url: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def database_url() -> str:
    """Rebuild the test schema via migrations, or skip if PostgreSQL is unreachable."""
    try:
        asyncio.run(asyncio.wait_for(_reset_schema(TEST_DATABASE_URL), timeout=5))
    except (OSError, TimeoutError, SQLAlchemyError) as exc:
        message = f"PostgreSQL not available at TEST_DATABASE_URL ({exc})"
        if os.environ.get("REQUIRE_TEST_DB"):  # CI: a missing database is a failure
            pytest.fail(message)
        pytest.skip(message)
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.attributes.update(
        database_url=TEST_DATABASE_URL,
        embedding_dim=TEST_EMBEDDING_DIM,
        configure_logger=False,
    )
    command.upgrade(config, "head")
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def settings(database_url: str, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=database_url,
        upload_dir=tmp_path_factory.mktemp("uploads"),
        embedding_dim=TEST_EMBEDDING_DIM,
        llm_provider="fake",
        embedding_provider="fake",
        fake_stream_delay_ms=0,
        chunk_size_tokens=120,
        chunk_overlap_tokens=20,
        log_level="WARNING",
    )


@pytest.fixture(scope="session")
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client for the app; every test starts from empty tables."""
    async with app.state.container.engine.begin() as conn:
        await conn.execute(text("TRUNCATE documents, conversations CASCADE"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
