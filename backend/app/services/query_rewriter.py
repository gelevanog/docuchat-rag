"""Turn follow-up questions into standalone retrieval queries using chat history."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from app.core.logging import get_logger
from app.core.text import content_terms, tokenize
from app.llm.base import ChatMessage, ChatModel, LLMError
from app.llm.prompts import REWRITE_SYSTEM_PROMPT, build_rewrite_messages

logger = get_logger(__name__)

_FOLLOW_UP_CUES = ("and ", "also ", "what about", "how about", "what if", "same ", "then ")
_MIN_STANDALONE_TERMS = 3
_MAX_REWRITE_CHARS = 500


class QueryRewriter(Protocol):
    async def rewrite(self, history: Sequence[ChatMessage], question: str) -> str: ...


class HeuristicQueryRewriter:
    """Offline rewriter: anchor short or cue-led follow-ups ("And for part-timers?",
    "What about Poland?") to the previous question; leave full questions alone."""

    async def rewrite(self, history: Sequence[ChatMessage], question: str) -> str:
        previous = next((m.content for m in reversed(history) if m.role == "user"), None)
        if previous is None:
            return question
        lowered = " ".join(tokenize(question)) + " "
        is_follow_up = lowered.startswith(_FOLLOW_UP_CUES)
        is_short = len(content_terms(question)) < _MIN_STANDALONE_TERMS
        if is_follow_up or is_short:
            return f"{question.strip()} (context: {previous.strip()})"
        return question


class LLMQueryRewriter:
    """Asks the chat model for a standalone query; falls back to the heuristic on failure."""

    def __init__(self, chat_model: ChatModel) -> None:
        self._chat_model = chat_model
        self._fallback = HeuristicQueryRewriter()

    async def rewrite(self, history: Sequence[ChatMessage], question: str) -> str:
        if not history:
            return question
        try:
            rewritten = await self._chat_model.complete(
                REWRITE_SYSTEM_PROMPT, build_rewrite_messages(history, question)
            )
        except LLMError as exc:
            logger.warning("query_rewrite_failed", error=str(exc))
            return await self._fallback.rewrite(history, question)
        cleaned = re.sub(r"\s+", " ", rewritten).strip().strip('"').strip()
        if not cleaned or len(cleaned) > _MAX_REWRITE_CHARS:
            return await self._fallback.rewrite(history, question)
        return cleaned


def build_query_rewriter(provider: str, chat_model: ChatModel) -> QueryRewriter:
    """The offline `fake` provider cannot paraphrase, so it uses the heuristic rewriter."""
    if provider == "fake":
        return HeuristicQueryRewriter()
    return LLMQueryRewriter(chat_model)
