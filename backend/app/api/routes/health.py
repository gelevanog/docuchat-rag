"""/health - liveness plus a database connectivity check."""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import __version__
from app.api.deps import ContainerDep, SessionDep
from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health")
async def health(container: ContainerDep, session: SessionDep, response: Response) -> HealthOut:
    try:
        await session.execute(text("SELECT 1"))
        database = "ok"
    except SQLAlchemyError:
        database = "unavailable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthOut(
        status="ok" if database == "ok" else "degraded",
        database=database,
        llm_provider=container.chat_model.name,
        embedding_provider=container.embedder.name,
        version=__version__,
    )
