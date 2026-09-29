"""Anthropic Claude chat model (embeddings come from EMBEDDING_PROVIDER)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Literal

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from pydantic import ValidationError

from app.core.logging import get_logger
from app.llm.base import ChatMessage, LLMError, SchemaT

logger = get_logger(__name__)

Effort = Literal["low", "medium", "high"]
REFUSAL_NOTICE = "\n\n_The model declined to answer this request._"
# Structured replies (grades, verdicts) are short. This leaves room for adaptive thinking
# while staying within the SDK's limit for non-streaming requests.
STRUCTURED_MAX_TOKENS = 16000


def _to_anthropic_messages(messages: Sequence[ChatMessage]) -> list[MessageParam]:
    return [{"role": message.role, "content": message.content} for message in messages]


class AnthropicChatModel:
    def __init__(self, client: AsyncAnthropic, model: str, max_tokens: int, effort: Effort) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=_to_anthropic_messages(messages),
                output_config={"effort": self._effort},
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        if final.stop_reason == "refusal":
            logger.warning("anthropic_refusal", model=self._model)
            yield REFUSAL_NOTICE
        elif final.stop_reason == "max_tokens":
            logger.warning("anthropic_max_tokens_reached", model=self._model)

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=_to_anthropic_messages(messages),
                output_config={"effort": "low"},
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("Anthropic declined the request")
        return "".join(block.text for block in response.content if block.type == "text")

    async def parse(
        self, system: str, messages: Sequence[ChatMessage], schema: type[SchemaT]
    ) -> SchemaT:
        try:
            response = await self._client.messages.parse(
                model=self._model,
                max_tokens=STRUCTURED_MAX_TOKENS,
                system=system,
                messages=_to_anthropic_messages(messages),
                output_format=schema,
                output_config={"effort": self._effort},
            )
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        except ValidationError as exc:  # truncated JSON or a value outside the schema's bounds
            raise LLMError(f"Anthropic reply does not match {schema.__name__}: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("Anthropic declined the request")
        if response.parsed_output is None:
            raise LLMError("Anthropic returned no structured output")
        return response.parsed_output
