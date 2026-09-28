"""Conversational RAG: rewrite -> retrieve -> generate (streamed) -> cite -> persist."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Conversation, Message, MessageRole
from app.llm.base import ChatMessage, ChatModel, LLMError
from app.llm.prompts import (
    ANSWER_SYSTEM_PROMPT,
    NO_ANSWER,
    build_answer_messages,
    parse_citations,
)
from app.retrieval.search import HybridRetriever
from app.retrieval.types import RetrievedChunk
from app.schemas import (
    ChatDoneEvent,
    ChatErrorEvent,
    ChatMetaEvent,
    ChatSourcesEvent,
    ChatTokenEvent,
    SourceOut,
)
from app.services.query_rewriter import QueryRewriter

logger = get_logger(__name__)

ChatEvent = tuple[str, BaseModel]

_TITLE_CHARS = 80
_SNIPPET_CHARS = 280


class ConversationNotFoundError(LookupError):
    pass


def make_snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def to_sources(chunks: Sequence[RetrievedChunk]) -> list[SourceOut]:
    return [
        SourceOut(
            id=index,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            filename=chunk.filename,
            page=chunk.page,
            heading=chunk.heading,
            snippet=make_snippet(chunk.content),
            content=chunk.content,
            score=round(chunk.score, 6),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def conversation_title(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= _TITLE_CHARS else title[: _TITLE_CHARS - 1].rstrip() + "…"


class ChatService:
    def __init__(
        self,
        chat_model: ChatModel,
        rewriter: QueryRewriter,
        retriever: HybridRetriever,
        default_top_k: int,
        history_turns: int,
    ) -> None:
        self._chat_model = chat_model
        self._rewriter = rewriter
        self._retriever = retriever
        self._default_top_k = default_top_k
        self._history_turns = history_turns

    async def get_or_create_conversation(
        self, session: AsyncSession, conversation_id: uuid.UUID | None, question: str
    ) -> Conversation:
        if conversation_id is not None:
            conversation = await session.get(Conversation, conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(str(conversation_id))
            return conversation
        conversation = Conversation(title=conversation_title(question))
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return conversation

    async def _history(
        self, session: AsyncSession, conversation_id: uuid.UUID
    ) -> list[ChatMessage]:
        if self._history_turns == 0:
            return []
        rows = await session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(self._history_turns * 2)
        )
        return [
            ChatMessage(
                role="user" if m.role == MessageRole.USER else "assistant", content=m.content
            )
            for m in reversed(list(rows))
        ]

    async def stream(
        self,
        session: AsyncSession,
        conversation: Conversation,
        question: str,
        document_ids: Sequence[uuid.UUID] | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[ChatEvent]:
        log = logger.bind(conversation_id=str(conversation.id))
        started = time.perf_counter()
        question = question.strip()

        history = await self._history(session, conversation.id)
        user_message = Message(
            conversation_id=conversation.id, role=MessageRole.USER, content=question
        )
        session.add(user_message)
        await session.flush()

        standalone = await self._rewriter.rewrite(history, question)
        user_message.rewritten_query = standalone if standalone != question else None
        await session.commit()
        yield (
            "meta",
            ChatMetaEvent(
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                rewritten_query=standalone,
            ),
        )

        chunks = await self._retriever.search(
            session, standalone, top_k or self._default_top_k, document_ids
        )
        sources = to_sources(chunks)
        yield "sources", ChatSourcesEvent(sources=sources)

        parts: list[str] = []
        if not chunks:
            parts.append(NO_ANSWER)
            yield "token", ChatTokenEvent(text=NO_ANSWER)
        else:
            prompt_question = question
            if standalone != question:
                prompt_question += f"\n(Interpreted as: {standalone})"
            messages = build_answer_messages(prompt_question, chunks, history)
            try:
                async for delta in self._chat_model.stream(ANSWER_SYSTEM_PROMPT, messages):
                    parts.append(delta)
                    yield "token", ChatTokenEvent(text=delta)
            except LLMError as exc:
                log.error("generation_failed", error=str(exc))
                yield "error", ChatErrorEvent(detail="The language model request failed.")
                return

        answer = "".join(parts).strip()
        cited_ids = parse_citations(answer, len(sources))
        citations = [sources[i - 1] for i in cited_ids]
        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            citations=[c.model_dump(mode="json") for c in citations],
        )
        session.add(assistant_message)
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation.id)
            .values(updated_at=func.now())
        )
        await session.commit()

        log.info(
            "chat_answered",
            model=self._chat_model.name,
            retrieved=len(chunks),
            cited=len(citations),
            rewritten=standalone != question,
            seconds=round(time.perf_counter() - started, 3),
        )
        yield (
            "done",
            ChatDoneEvent(message_id=assistant_message.id, answer=answer, citations=citations),
        )
