"""/api/documents - upload, list, inspect and delete documents."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, UploadFile, status

from app.api.deps import ContainerDep, SessionDep
from app.db.models import Document
from app.schemas import DocumentOut, DocumentUploadOut
from app.services.documents import UploadRejectedError

router = APIRouter(prefix="/api/documents", tags=["documents"])


async def _get_or_404(
    container: ContainerDep, session: SessionDep, document_id: uuid.UUID
) -> Document:
    document = await container.documents.get(session, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return document


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document (PDF, DOCX, Markdown or TXT)",
    responses={
        413: {"description": "File too large"},
        415: {"description": "Unsupported or empty file"},
    },
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    container: ContainerDep,
    session: SessionDep,
) -> DocumentUploadOut:
    """Stores the file and schedules ingestion in the background.

    Poll `GET /api/documents/{id}` until `status` is `ready` (or `failed`).
    Uploading identical content again returns the existing document with `duplicate=true`.
    """
    limit = container.settings.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"The file exceeds the {container.settings.max_upload_mb} MB limit",
        )
    try:
        outcome = await container.documents.upload(session, file.filename or "upload", data)
    except UploadRejectedError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc

    if outcome.needs_ingestion:
        background_tasks.add_task(container.pipeline.ingest, outcome.document.id)
    return DocumentUploadOut(
        document=DocumentOut.model_validate(outcome.document), duplicate=outcome.duplicate
    )


@router.get("", summary="List documents")
async def list_documents(container: ContainerDep, session: SessionDep) -> list[DocumentOut]:
    documents = await container.documents.list_documents(session)
    return [DocumentOut.model_validate(document) for document in documents]


@router.get("/{document_id}", summary="Get a document and its ingestion status")
async def get_document(
    document_id: uuid.UUID, container: ContainerDep, session: SessionDep
) -> DocumentOut:
    return DocumentOut.model_validate(await _get_or_404(container, session, document_id))


@router.delete(
    "/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document"
)
async def delete_document(
    document_id: uuid.UUID, container: ContainerDep, session: SessionDep
) -> Response:
    document = await _get_or_404(container, session, document_id)
    await container.documents.delete(session, document)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
