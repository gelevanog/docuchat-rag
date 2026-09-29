"""The golden question set and document ingestion shared by the evaluation scripts."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.container import Container
from app.db.models import DocumentStatus
from app.ingestion.parsers import SUPPORTED_EXTENSIONS


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    question: str
    expected_document: str
    """Filename of the document that answers the question (document-level hit)."""
    expected_text: str
    """A phrase the answering chunk must contain (chunk-level hit)."""
    tags: tuple[str, ...] = ()


def load_questions(path: Path, tag: str | None = None) -> list[GoldenQuestion]:
    """Read a golden set; with `tag`, keep only the questions carrying that tag."""
    questions = [
        GoldenQuestion(**{**item, "tags": tuple(item.get("tags", ()))})
        for item in yaml.safe_load(path.read_text())["questions"]
    ]
    return [q for q in questions if tag is None or tag in q.tags]


def read_documents(docs_dir: Path) -> list[tuple[str, bytes]]:
    return [
        (path.name, path.read_bytes())
        for path in sorted(docs_dir.iterdir())
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


async def ingest_directory(container: Container, docs_dir: Path) -> dict[str, uuid.UUID]:
    """Upload and ingest every supported file (a no-op for content already ingested);
    returns filename -> document id."""
    ids: dict[str, uuid.UUID] = {}
    for name, data in await asyncio.to_thread(read_documents, docs_dir):
        async with container.sessionmaker() as session:
            outcome = await container.documents.upload(session, name, data)
        if outcome.needs_ingestion or outcome.document.status != DocumentStatus.READY:
            await container.pipeline.ingest(outcome.document.id)
        ids[outcome.document.filename] = outcome.document.id
    return ids
