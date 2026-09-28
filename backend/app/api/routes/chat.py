"""/api/chat - ask a question and stream a cited answer over Server-Sent Events."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.api.deps import ContainerDep, SessionDep
from app.db.models import Conversation
from app.schemas import ChatRequest
from app.services.chat import ConversationNotFoundError

router = APIRouter(prefix="/api", tags=["chat"])


async def resolve_conversation(
    payload: ChatRequest, container: ContainerDep, session: SessionDep
) -> Conversation:
    """Runs before the stream starts, so an unknown conversation is a proper 404."""
    try:
        return await container.chat.get_or_create_conversation(
            session, payload.conversation_id, payload.message
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found") from exc


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    summary="Ask a question (streams SSE events)",
    responses={404: {"description": "Conversation not found"}},
)
async def chat(
    payload: ChatRequest,
    conversation: Annotated[Conversation, Depends(resolve_conversation)],
    container: ContainerDep,
    session: SessionDep,
) -> AsyncIterable[ServerSentEvent]:
    """Streams events in this order:

    * `meta` - `{conversation_id, user_message_id, rewritten_query}`
    * `sources` - `{sources: [...]}` the retrieved chunks, numbered like the citations
    * `token` - `{text}` answer deltas (many)
    * `done` - `{message_id, answer, citations}` only the sources the answer cites
    * `error` - `{detail}` if generation fails
    """
    async for event, data in container.chat.stream(
        session,
        conversation,
        payload.message,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
    ):
        yield ServerSentEvent(event=event, data=data)
