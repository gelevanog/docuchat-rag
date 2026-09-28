"""/api/conversations - chat history."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import SessionDep
from app.db.models import Conversation
from app.schemas import ConversationDetailOut, ConversationOut

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", summary="List conversations, most recent first")
async def list_conversations(
    session: SessionDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[ConversationOut]:
    result = await session.scalars(
        select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
    )
    return [ConversationOut.model_validate(conversation) for conversation in result]


@router.get("/{conversation_id}", summary="Get a conversation with its messages")
async def get_conversation(
    conversation_id: uuid.UUID, session: SessionDep
) -> ConversationDetailOut:
    conversation = await session.scalar(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return ConversationDetailOut.model_validate(conversation)


@router.delete(
    "/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a conversation"
)
async def delete_conversation(conversation_id: uuid.UUID, session: SessionDep) -> Response:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    await session.delete(conversation)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
