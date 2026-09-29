"""Pydantic models shared by the HTTP API and the services (request/response + SSE payloads)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import DocumentStatus, MessageRole


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- Documents -------------------------------------------------------------------


class DocumentOut(ORMModel):
    id: uuid.UUID
    filename: str
    title: str
    file_type: str
    size_bytes: int
    status: DocumentStatus
    error: str | None
    page_count: int | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class DocumentUploadOut(BaseModel):
    document: DocumentOut
    duplicate: bool = Field(
        description="True if identical content was already uploaded (the upload was a no-op)."
    )


# --- Chat ------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1, max_length=4000, examples=["How many vacation days do I get?"]
    )
    conversation_id: uuid.UUID | None = Field(
        default=None, description="Continue an existing conversation; omit to start a new one."
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None, description="Restrict retrieval to these documents."
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Number of chunks to retrieve."
    )


class SourceOut(BaseModel):
    """A retrieved chunk, numbered as it was shown to the model (`[id]` in the answer)."""

    id: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    filename: str
    page: int | None
    heading: str | None
    snippet: str
    content: str
    score: float = Field(description="Reciprocal Rank Fusion score of the hybrid search.")
    rerank_score: float | None = Field(
        default=None, description="Re-ranker relevance in [0, 1]; null when re-ranking is off."
    )


class ChatMetaEvent(BaseModel):
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    rewritten_query: str


class ChatSourcesEvent(BaseModel):
    sources: list[SourceOut]


class ChatTokenEvent(BaseModel):
    text: str


class ChatDoneEvent(BaseModel):
    message_id: uuid.UUID
    answer: str
    citations: list[SourceOut] = Field(description="Only the sources the answer actually cites.")


class ChatErrorEvent(BaseModel):
    detail: str


# --- Conversations ---------------------------------------------------------------


class MessageOut(ORMModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    rewritten_query: str | None
    citations: list[SourceOut] | None
    created_at: datetime


class ConversationOut(ORMModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut]


# --- Health ----------------------------------------------------------------------


class HealthOut(BaseModel):
    status: str
    database: str
    llm_provider: str
    embedding_provider: str
    reranker: str | None
    version: str
