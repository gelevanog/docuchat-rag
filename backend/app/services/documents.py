"""Document upload, deduplication, listing and deletion."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePath

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Document, DocumentStatus
from app.ingestion.parsers import detect_file_type

logger = get_logger(__name__)


class UploadRejectedError(ValueError):
    """The upload is empty, too large, or of an unsupported type."""


@dataclass(frozen=True, slots=True)
class UploadOutcome:
    document: Document
    duplicate: bool
    """True when a file with identical content already existed."""
    needs_ingestion: bool


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DocumentService:
    def __init__(self, upload_dir: Path, max_upload_bytes: int) -> None:
        self._upload_dir = upload_dir
        self._max_upload_bytes = max_upload_bytes

    async def upload(self, session: AsyncSession, filename: str, data: bytes) -> UploadOutcome:
        filename = PurePath(filename).name  # drop any client-supplied directories
        if not data:
            raise UploadRejectedError("The file is empty")
        if len(data) > self._max_upload_bytes:
            raise UploadRejectedError(
                f"The file exceeds the {self._max_upload_bytes // (1024 * 1024)} MB limit"
            )
        try:
            file_type = detect_file_type(filename)
        except ValueError as exc:
            raise UploadRejectedError(str(exc)) from exc

        digest = content_hash(data)
        existing = await self._get_by_hash(session, digest)
        if existing is not None:
            return await self._outcome_for_existing(session, existing)

        document_id = uuid.uuid4()
        path = self._upload_dir / f"{document_id}{PurePath(filename).suffix.lower()}"
        await asyncio.to_thread(self._write_file, path, data)
        document = Document(
            id=document_id,
            filename=filename,
            title=PurePath(filename).stem,
            file_type=file_type,
            content_hash=digest,
            size_bytes=len(data),
            storage_path=str(path),
            status=DocumentStatus.PENDING,
        )
        session.add(document)
        try:
            await session.commit()
        except IntegrityError:
            # A concurrent upload of the same content won the race.
            await session.rollback()
            await asyncio.to_thread(path.unlink, missing_ok=True)
            existing = await self._get_by_hash(session, digest)
            if existing is None:
                raise
            return await self._outcome_for_existing(session, existing)

        await session.refresh(document)
        logger.info("document_uploaded", document_id=str(document_id), filename=filename)
        return UploadOutcome(document=document, duplicate=False, needs_ingestion=True)

    async def list_documents(self, session: AsyncSession) -> list[Document]:
        result = await session.scalars(select(Document).order_by(Document.created_at.desc()))
        return list(result)

    async def get(self, session: AsyncSession, document_id: uuid.UUID) -> Document | None:
        return await session.get(Document, document_id)

    async def delete(self, session: AsyncSession, document: Document) -> None:
        path = Path(document.storage_path)
        await session.delete(document)
        await session.commit()
        await asyncio.to_thread(path.unlink, missing_ok=True)
        logger.info("document_deleted", document_id=str(document.id))

    @staticmethod
    async def _get_by_hash(session: AsyncSession, digest: str) -> Document | None:
        return await session.scalar(select(Document).where(Document.content_hash == digest))

    @staticmethod
    async def _outcome_for_existing(session: AsyncSession, document: Document) -> UploadOutcome:
        """Re-uploading identical content is a no-op, unless the previous attempt failed."""
        if document.status != DocumentStatus.FAILED:
            return UploadOutcome(document=document, duplicate=True, needs_ingestion=False)
        document.status = DocumentStatus.PENDING
        document.error = None
        await session.commit()
        await session.refresh(document)
        return UploadOutcome(document=document, duplicate=True, needs_ingestion=True)

    @staticmethod
    def _write_file(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
